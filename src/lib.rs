//! A branded graph type.
//!
//! See the documentation of [`Graph`] for more details.
use std::fmt::{self, Debug, Display, Formatter};
use std::marker::PhantomData;
use std::ops::{Index, IndexMut};

/// A marker type, which makes `'id` invariant and is zero-sized.
///
/// It uses a function type instead of `*mut &'id ()`, as in the GhostCell
/// paper, so that it can be `Send + Sync`, while still having an invariant
/// lifetime.
type InvariantLifetime<'id> = PhantomData<fn(&'id ()) -> &'id ()>;

/// An opaque, owned graph structure, that must be unlocked to be accessed or
/// modified.
///
/// See the documentation of [`Graph`] for more details.
#[repr(transparent)]
#[derive(PartialEq, Eq)]
pub struct BrandedGraph {
    inner: Graph<'static>,
}

/// A branded graph structure with safe and fast self-references.
///
/// A `Graph` can only be obtained as a `&` or `&mut` reference via a
/// [`BrandedGraph`] by [`unlock`](BrandedGraph::unlock) or [`unlock_mut`](BrandedGraph::unlock_mut).
///
/// A node is referenced through a [`NodeRef<'id>`](NodeRef) index, which is
/// branded with an invariant lifetime, `'id`, tied to the `Graph` that created
/// it. This ensures that references cannot be used with a graph other than the
/// one they were created with. Since nodes can only be pushed (i.e., the length
/// is monotonically increasing), indexing into the graph can safely be
/// [unchecked](slice::get_unchecked). `Graph` builds on the [`BrandedVec`](https://matyama.github.io/rust-examples/rust_examples/brands/index.html)
/// technique from [“GhostCell: Separating Permissions from Data in Rust”](https://plv.mpi-sws.org/rustbelt/ghostcell/)
/// (Yanovski et al., 2021).
///
/// In combination, it uses a flat arena structure with indexed references,
/// instead of storing nodes in `Rc`. For more about flattening graphs, read
/// [Adrian Sampson's explanation](https://www.cs.cornell.edu/~asampson/blog/flattening.html)
/// of the technique.
///
/// # Example
///
/// A graph is modified within `unlock` or `unlock_mut`:
///
/// ```
/// # use branded_graph::*;
/// let mut g = BrandedGraph::new();
/// g.unlock_mut(|g: &mut Graph<'_>| {
///     let x = g.push(Node::Number(1));
///     let y = g.push(Node::Number(2));
///     g.push(Node::Add(x, y));
///     println!("{g:#?}");
/// });
/// ```
///
/// # Safety
///
/// The API prevents any safety errors and should be sound. However, it has not
/// been proven like in the GhostCell paper. This technique uses lifetimes in an
/// unintended way, so the type errors from violating invariants are not
/// intuitive.
///
/// An index may only be used by the graph that created it:
///
/// ```compile_fail
/// # use branded_graph::*;
/// let mut g1 = BrandedGraph::new();
/// let mut g2 = BrandedGraph::new();
/// g1.unlock_mut(|g1| {
///     g2.unlock_mut(|g2| {
///         let x = g1.push(Node::Number(42));
///         println!("{}", g1[x]); // ok
///         println!("{}", g2[x]); // error
///     });
/// });
/// ```
///
/// Nodes cannot be constructed, that reference nodes from another graph:
///
/// ```compile_fail
/// # use branded_graph::*;
/// # let mut g1 = BrandedGraph::new();
/// # let mut g2 = BrandedGraph::new();
/// # g1.unlock_mut(|g1| {
/// #     g2.unlock_mut(|g2| {
/// // …
///         let x = g1.push(Node::Number(1));
///         let y = g1.push(Node::Number(2));
///         g2.push(Node::Add(x, y)); // error
/// #     });
/// # });
/// ```
///
/// Graphs cannot be swapped:
///
/// ```compile_fail
/// # use branded_graph::*;
/// # let mut g1 = BrandedGraph::new();
/// # let mut g2 = BrandedGraph::new();
/// # g1.unlock_mut(|g1| {
/// #     g2.unlock_mut(|g2| {
/// // …
///         std::mem::replace(&mut g1, g2); // error
/// #     });
/// # });
/// ```
///
/// Nodes between graphs cannot be compared:
///
/// ```compile_fail
/// # use branded_graph::*;
/// # let mut g1 = BrandedGraph::new();
/// # let mut g2 = BrandedGraph::new();
/// # g1.unlock_mut(|g1| {
/// #     g2.unlock_mut(|g2| {
/// // …
///         let x = g1.push(Node::Number(42));
///         let y = g2.push(Node::Number(42));
///         println!("equal? {}", x == y); // error
/// #     })
/// # });
/// ```
///
/// The graph, nodes, and node references cannot be smuggled out:
///
/// ```compile_fail
/// # use branded_graph::*;
/// let mut g = BrandedGraph::new();
/// let node = g.unlock_mut(|g| g.push(Node::Number(42))); // error
/// ```
///
/// These types are all `Send + Sync`:
///
/// ```
/// # use branded_graph::*;
/// # use static_assertions::assert_impl_all;
/// assert_impl_all!(BrandedGraph: Send, Sync);
/// assert_impl_all!(Graph: Send, Sync);
/// assert_impl_all!(NodeRef: Send, Sync);
/// assert_impl_all!(Node: Send, Sync);
/// ```
#[repr(transparent)]
#[derive(PartialEq, Eq)]
pub struct Graph<'id> {
    nodes: Vec<Node<'id>>,
    _marker: InvariantLifetime<'id>,
}

/// A reference to a node in the graph with lifetime `'id`.
///
/// It is a wrapper over a `u32`, which is sufficient for any practical graph
/// size.
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NodeRef<'id> {
    index: u32,
    _marker: InvariantLifetime<'id>,
}

/// A node in the graph with lifetime `'id`.
///
/// This is a very minimal set of expression kinds, but can be extended.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Node<'id> {
    Number(i64),
    Add(NodeRef<'id>, NodeRef<'id>),
}

impl BrandedGraph {
    pub fn new() -> Self {
        BrandedGraph {
            inner: Graph {
                nodes: Vec::new(),
                _marker: InvariantLifetime::default(),
            },
        }
    }

    /// Obtain a reference to the inner `&Graph<'id>`.
    pub fn unlock<T>(&self, f: impl for<'a> FnOnce(&Graph<'a>) -> T) -> T {
        f(&self.inner)
    }

    /// Obtain a reference to the inner `&mut Graph<'id>`.
    pub fn unlock_mut<T>(&mut self, f: impl for<'a> FnOnce(&mut Graph<'a>) -> T) -> T {
        f(&mut self.inner)
    }
}

impl<'id> Graph<'id> {
    /// Push a node and return a reference to it, branded to this graph.
    pub fn push(&mut self, node: Node<'id>) -> NodeRef<'id> {
        let index = self.nodes.len();
        self.nodes.push(node);
        NodeRef::new(index)
    }

    /// Create a branded reference to the node at the given index.
    pub fn get_index(&self, index: usize) -> Option<NodeRef<'id>> {
        if index < self.nodes.len() {
            Some(NodeRef::new(index))
        } else {
            None
        }
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn iter(&self) -> impl Iterator<Item = &Node<'id>> {
        self.nodes.iter()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut Node<'id>> {
        self.nodes.iter_mut()
    }

    pub fn iter_entries(&self) -> impl Iterator<Item = (NodeRef<'id>, &Node<'id>)> {
        self.nodes
            .iter()
            .enumerate()
            .map(|(i, node)| (NodeRef::new(i), node))
    }

    pub fn iter_entries_mut(&mut self) -> impl Iterator<Item = (NodeRef<'id>, &mut Node<'id>)> {
        self.nodes
            .iter_mut()
            .enumerate()
            .map(|(i, node)| (NodeRef::new(i), node))
    }

    pub fn iter_refs(&self) -> impl Iterator<Item = NodeRef<'id>> {
        (0..self.len()).map(|i| NodeRef::new(i))
    }
}

impl<'id> NodeRef<'id> {
    fn new(index: usize) -> Self {
        NodeRef {
            index: index as u32,
            _marker: InvariantLifetime::default(),
        }
    }

    pub fn index(&self) -> usize {
        self.index as usize
    }
}

impl<'id> Index<NodeRef<'id>> for Graph<'id> {
    type Output = Node<'id>;

    fn index(&self, index: NodeRef<'id>) -> &Node<'id> {
        unsafe { self.nodes.get_unchecked(index.index()) }
    }
}

impl<'id> IndexMut<NodeRef<'id>> for Graph<'id> {
    fn index_mut(&mut self, index: NodeRef<'id>) -> &mut Node<'id> {
        unsafe { self.nodes.get_unchecked_mut(index.index()) }
    }
}

impl Clone for BrandedGraph {
    fn clone(&self) -> Self {
        BrandedGraph {
            inner: Graph {
                nodes: self.inner.nodes.clone(),
                _marker: InvariantLifetime::default(),
            },
        }
    }

    fn clone_from(&mut self, source: &Self) {
        self.inner.nodes.clone_from(&source.inner.nodes)
    }
}

impl Debug for BrandedGraph {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str("Branded")?;
        Debug::fmt(&self.inner, f)
    }
}

impl Debug for Graph<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str("Graph ")?;
        if f.alternate() {
            struct DisplayDebug<T: Display>(T);
            impl<T: Display> Debug for DisplayDebug<T> {
                fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
                    Display::fmt(&self.0, f)
                }
            }

            f.debug_map()
                .entries(
                    self.iter_entries()
                        .map(|(i, node)| (DisplayDebug(i), DisplayDebug(node))),
                )
                .finish()
        } else {
            f.debug_map()
                .entries(self.nodes.iter().enumerate())
                .finish()
        }
    }
}

impl Debug for NodeRef<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_tuple("NodeRef").field(&self.index).finish()
    }
}

impl Display for Node<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Node::Number(n) => write!(f, "number {n}"),
            Node::Add(lhs, rhs) => write!(f, "add {lhs} {rhs}"),
        }
    }
}

impl Display for NodeRef<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "%{}", self.index)
    }
}
