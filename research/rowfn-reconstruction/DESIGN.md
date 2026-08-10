<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# RowFn design

## Problem statement

A scalar function receives arrays, but its mathematical definition often describes one row. For
example, checked addition has this row definition:

```rust
fn checked_add(lhs: i64, rhs: i64) -> (i64, bool) {
    lhs.overflowing_add(rhs)
}
```

A complete array implementation also needs to do this work:

- Validate both dtypes.
- Decode both arrays into representations with cheap row access.
- Preserve or collapse batch constants.
- Combine input validity.
- Select dense or valid-only execution.
- Allocate output.
- Attribute failures only to valid rows.
- Build an array with the declared dtype and length.

RowFn keeps the row definition small and implements the column concerns once.

## Public declaration

A row function declares its options, argument names, identity, fallibility, and dtype dispatch.
The essential trait has this shape:

```rust
trait RowFn: Clone + Send + Sync + 'static {
    type Options;

    const ARG_NAMES: &'static [&'static str];
    const FALLIBLE: bool = false;

    fn id(&self) -> ScalarFnId;

    fn dispatch<V: RowVisitor>(
        &self,
        options: &Self::Options,
        args: &[DType],
        visitor: V,
    ) -> VortexResult<V::VisitResult>;

    fn reduce_encoded(
        &self,
        options: &Self::Options,
        args: &[ArrayRef],
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>>;
}
```

`dispatch` selects concrete Rust element types. Planning and execution call the same method with
different visitor types. Therefore, `dispatch` must select the same visit from only `options` and
the input dtypes.

The `reduce_encoded` hook is optional. It gives an encoding-aware implementation the original
arrays before row decoding. `None` selects the row loop.

## Why dispatch uses a visitor

The return type of a generic visit depends on whether the caller plans or executes. Stable Rust
cannot return one closure with caller-selected generic types from a normal function. The visitor
reverses control:

```text
ScalarFnVTable::return_dtype
    -> RowFn::dispatch(PlanRows)
        -> visitor.visit::<ConcreteArgs, ConcreteOutput>(closure)
            -> BatchPlan

ScalarFnVTable::execute
    -> RowFn::dispatch(ExecuteRows)
        -> visitor.visit::<ConcreteArgs, ConcreteOutput>(closure)
            -> RowExecution
```

The function chooses `ConcreteArgs` and `ConcreteOutput`. The framework chooses what a visit does.
The compiler monomorphizes both paths for those concrete types.

The planning visitor does not call the row closure. It validates the selected input and output
types, checks compile-time contracts, and selects a null policy. The execution visitor decodes the
arrays and runs the matching loop.

## Visit capabilities

The visitor has six entry points.

| Method | Output model | Row error model | Preparation |
| --- | --- | --- | --- |
| `visit` | Independent owned value | None | None |
| `visit_prepared` | Independent owned value | None | Once per batch |
| `visit_deferred` | Independent owned value | OR-reduced evidence | None |
| `visit_prepared_deferred` | Independent owned value | OR-reduced evidence | Once per batch |
| `visit_into` | Sink row handle | `SinkResult` | None |
| `visit_prepared_into` | Sink row handle | `SinkResult` | Once per batch |

The unprepared methods exist for the common case:

```rust
visitor.visit::<(i64, i64), i64>(|(lhs, rhs)| lhs.wrapping_add(rhs))
```

Their default implementation supplies an empty prepared value:

```rust
self.visit_prepared::<Args, Out, ()>(
    |_| (),
    move |&(), args| apply(args),
)
```

This delegation keeps planning and execution logic in the prepared methods only.

## Input elements

`InputElement` connects one logical Rust row value to one decoded array representation:

```rust
trait InputElement {
    type Column;
    type Varying<'a>;
    type Elem<'a>;

    const DENSE_SAFE: bool;
    const DECODE_FALLIBLE: bool;

    fn validate(dtype: &DType) -> VortexResult<()>;
    fn decode(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self::Column>;
    fn get(column: &Self::Column, index: usize) -> Self::Elem<'_>;
    fn varying(column: &Self::Column) -> Self::Varying<'_>;
    fn varying_len(column: &Self::Varying<'_>) -> usize;
    unsafe fn get_varying_unchecked(
        column: &Self::Varying<'_>,
        index: usize,
    ) -> Self::Elem<'_>;
}
```

`Column` owns the decoded batch representation. `Varying` is the cheaper view used by an
all-varying loop. `Elem` is the value that the row closure receives.

For `i64`, these types are:

```rust
type Column = Buffer<i64>;
type Varying<'a> = &'a [i64];
type Elem<'a> = i64;
```

The decode step performs the array execution and ptype downcast once. The row loop sees a slice
and `i64` values. It does not see `ArrayRef`, a trait object, a ptype match, or an execution
context.

For a tensor row of `f32`, these types are:

```rust
type Column = TensorRows<f32>;
type Varying<'a> = &'a TensorRows<f32>;
type Elem<'a> = &'a [f32];
```

`TensorRows` stores one typed flat buffer, the row count, the width, and a stride. The row access
computes one offset and returns a slice. This removes a ptype check and buffer downcast from every
row.

## Concrete `Args::varying` examples

`ElementTuple` combines input elements. It decodes each input into an `ArgColumn`:

```rust
enum ArgColumnKind<T: InputElement> {
    Varying(T::Column),
    Constant(T::Column),
}
```

A constant column is decoded as one physical row. The logical batch length stays separate.

### Example 1: column plus column

Consider this logical input:

```text
lhs = [10, 20, 30]
rhs = [ 1,  2,  3]
```

The decoded tuple is conceptually:

```text
columns = (
    Varying(Buffer([10, 20, 30])),
    Varying(Buffer([1, 2, 3])),
)
```

`Args::varying(&columns)` asks both arguments for direct varying views:

```rust
Some((
    columns.0.varying()?,
    columns.1.varying()?,
))
```

Both calls return `Some`, so the result is:

```text
Some((&[10, 20, 30], &[1, 2, 3]))
```

The executor validates both lengths once. It then creates a `LaneZip` source. The source yields:

```text
index 0 -> (10, 1)
index 1 -> (20, 2)
index 2 -> (30, 3)
```

The hot loop does not inspect `ArgColumnKind`.

### Example 2: column plus constant

Now consider this logical input:

```text
lhs = [10, 20, 30]
rhs = Constant(7, logical_len = 3)
```

The decoded tuple is conceptually:

```text
columns = (
    Varying(Buffer([10, 20, 30])),
    Constant(Buffer([7])),
)
```

The first `varying()?` succeeds. The second returns `None`. The `?` returns `None` from the tuple
method, so this is the result:

```text
Args::varying(&columns) == None
```

`None` does not mean that no input varies. It means that the tuple is not _all varying_. The mixed
loop uses `Args::get`:

```text
index 0 -> (columns.0[0], columns.1[0]) -> (10, 7)
index 1 -> (columns.0[1], columns.1[0]) -> (20, 7)
index 2 -> (columns.0[2], columns.1[0]) -> (30, 7)
```

This loop performs one `ArgColumnKind` match for each argument and row. It avoids allocating or
expanding `[7, 7, 7]`.

The preparation input is independent from `Args::varying`:

```text
Args::constants(&columns) == (None, Some(7))
```

A prepared closure can precompute work from `7`. An ordinary closure can ignore the preparation
input and still use the mixed loop.

### Example 3: constant plus constant

If both inputs are non-null constants, batch execution takes a higher-level fast path. It executes
one row and broadcasts the result to the logical batch length.

The row executor can still represent two constants. This representation matters for a masked
constant because the strict validity can prevent the all-constant broadcast path.

## Why `Args::varying` exists

The simplest loop can call `Args::get` for every input shape. That loop contains a branch for each
argument and row:

```rust
for index in 0..row_count {
    let lhs = match lhs_column {
        Varying(values) => values[index],
        Constant(value) => value[0],
    };
    let rhs = match rhs_column {
        Varying(values) => values[index],
        Constant(value) => value[0],
    };
    output[index] = apply(lhs, rhs);
}
```

For two varying arrays, these branches always choose the same arm. `Args::varying` selects that
shape once before the loop. The all-varying loop then contains only loads, arithmetic, failure
reduction, and stores.

`VaryingColumns` also removes buffer descriptors from the row path. A primitive tuple becomes two
slices, and a `LaneZip` gives LLVM independent indexed loads.

## Owned output

`OutputElement` describes a Rust value that builds an all-valid array:

```rust
trait OutputElement {
    fn element_dtype() -> DType;
    fn build(values: Vec<Self>) -> ArrayRef;
}
```

The dtype cannot depend on runtime input metadata. Primitive output fits this model. A tensor
output whose shape comes from an input dtype does not.

The owned executor allocates `Vec<Out>` once. It exposes the spare capacity as
`[MaybeUninit<Out>]`. The loop writes each row directly into its final output slot.

The vector length remains zero until the loop finishes. Therefore, an unwind does not drop
uninitialized slots. A compile-time assertion rejects output types that require drop glue. After
normal completion, the executor sets the length once and builds the array.

## Output sinks

An output sink supports runtime-shaped output and shared batch state:

```rust
trait OutputSink {
    type Rows<'a>;
    type Row<'a>;
    type WriteToken;

    fn with_capacity(rows: usize, dtype: &DType) -> VortexResult<Self>;
    fn rows(&mut self) -> Self::Rows<'_>;
    fn row(rows: &mut Self::Rows<'_>, index: usize) -> Self::Row<'_>;
    fn finish(self, error: DeferredError) -> VortexResult<ArrayRef>;
}
```

The executor borrows `Rows` once before the loop. This keeps the sink descriptor and shape as loop
invariants. The closure receives only the row handle.

`OutputSink::WriteToken` ties each sink to the result from its row closure. Initialized sinks use
`()`. `UninitElementSink<T>` requires `InitializedElement` and exposes each row as
`&mut MaybeUninit<T>`:

```rust
visitor.visit_into::<Args, UninitElementSink<T>, _>(|args, output| {
    let value = apply(args);

    // SAFETY: `output` is the `UninitElementSink` row supplied for this callback.
    unsafe { InitializedElement::write(output, value) }
})
```

`InitializedElement` is zero-sized write evidence. Only unsafe code can construct it. The caller
must write the current callback's row and return the token from that callback. The sink calls
`Vec::set_len` only after every successful row returns this evidence. A valid-only loop initializes
placeholders before it skips rows.

## Failure models

An immediate `VortexResult` leaves the loop on the first error. This model is appropriate when the
operation is expensive and scalar, such as integer division.

Deferred failure separates cheap row evidence from expensive error construction:

```rust
let mut failed = Fail::default();
for index in 0..row_count {
    let (value, row_failure) = apply(input[index]);
    failed |= row_failure;
    output[index].write(value);
}
finish_failure(failed)
```

The failure type must be no wider than the output type. A wide loop-carried reduction can limit
the vector width. The default failure value must mean success, including for an empty batch.

The closure creates no `VortexError`. A cold function creates the rich error after the loop.

## Why `RowExecution` exists

Dense execution can evaluate stored payloads behind null rows. A checked operation can report a
failure from such a payload. That failure must not escape if the logical row is null.

`RowExecution` preserves this distinction:

```rust
enum RowExecution {
    Output(ArrayRef),
    DeferredError(VortexError),
}
```

An outer `VortexResult` carries immediate or structural errors. `DeferredError` means that the loop
finished and produced only retryable failure evidence.

For mixed validity, batch execution filters to valid rows and repeats the dense loop. The second
result decides whether the error is observable. Once a path contains only valid rows,
`From<RowExecution> for VortexResult<ArrayRef>` turns a deferred error into an ordinary error.

## Null execution policies

Planning derives one policy from the concrete input and result types.

### `Dense`

This policy applies when decoding and the closure tolerate all stored null payloads. The kernel
visits every row and batch execution masks the output.

Primitive arithmetic uses this policy when it is infallible. A null primitive row still stores a
valid Rust primitive value, although that value is logically unspecified.

### `DenseWithRetry`

This policy applies to dense-safe inputs with deferred failure evidence. The first loop visits all
rows. If it reports failure, batch execution materializes validity and retries only valid rows.

This policy preserves the fast dense loop for the common success case. It also prevents a null
payload from creating an observable error.

### `ValidOnly`

This policy applies when decoding or row access cannot tolerate null payloads. Batch execution
first asks the sink to skip invalid rows over the original arrays. If the input or sink cannot
support that path, batch execution filters every input and scatters the compact result.

Geometry uses this policy. Some geometry encodings can decode a harmless placeholder for null
rows. The loop then reads only the valid indices.

## Prepared constants

A prepared visit receives `Option<Elem>` for each argument before the row loop. `Some` means that
the argument is a batch constant.

Cosine similarity uses this capability to compute a constant operand norm once:

```rust
prepare((lhs, rhs)) -> ConstNorms {
    lhs: lhs.map(l2_norm_row),
    rhs: rhs.map(l2_norm_row),
}
```

Each row still computes its inner product. It reuses a prepared norm when an operand is constant.

Spatial containment and intersection use the same pattern for constant geometry metadata and
bounding boxes. The preparation step removes repeated work without adding a specialized array
kernel.

## Loop shape that LLVM receives

For an all-varying primitive pair, monomorphization reduces the framework to this essential loop:

```rust
let mut failed = Fail::default();
for index in 0..len {
    let lhs = unsafe { *lhs.get_unchecked(index) };
    let rhs = unsafe { *rhs.get_unchecked(index) };
    let (value, row_failure) = apply((lhs, rhs));
    failed |= row_failure;
    unsafe { output.get_unchecked_mut(index).write(value) };
}
```

The loop has these properties:

- The input and output element types are concrete.
- The closure is concrete and inlineable.
- Input lengths are equal and validated before the loop.
- The output length equals the input length.
- Each iteration reads and writes an independent index.
- The failure reduction is associative bitwise OR.
- Rich errors, array construction, dtype dispatch, and validity logic are outside the loop.

These properties make the loop suitable for LLVM autovectorization. They do not force LLVM to use
SIMD for every operation.

## Compile-time contracts

Const assertions reject these invalid declarations during compilation:

- The element tuple arity differs from `RowFn::ARG_NAMES`.
- Input decoding can fail, but `RowFn::FALLIBLE` is false.
- A row result can fail, but `RowFn::FALLIBLE` is false.
- An owned output requires drop glue.
- Deferred failure evidence is wider than the output.
- A sink and its result disagree about deferred errors.

Runtime planning validates input dtypes and output nullability. Batch finalization validates output
length and dtype.
