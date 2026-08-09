#set page(paper: "a4", margin: 2.2cm, numbering: "1 / 1")
#set text(font: "Libertinus Serif", size: 10.5pt)
#set par(justify: true, leading: 0.62em)
#set heading(numbering: "1.")
#show heading: it => block(above: 1.4em, below: 0.8em, it)
#show raw: it => text(font: "Noto Sans Mono", size: 0.88em, it)
#set table(stroke: 0.4pt + luma(65%), inset: 5pt)

#let mask = math.op("mask")
#let valid = math.op("valid")
#let N = text(fill: rgb("#b03a2e"), weight: "bold", [NULL])

#let node(body, fill: luma(96%)) = box(
  inset: (x: 7pt, y: 5pt), radius: 3pt, stroke: 0.5pt + luma(55%), fill: fill, body,
)

#let lead(body) = block(
  inset: (x: 10pt, y: 8pt), radius: 3pt, fill: luma(97%),
  stroke: (left: 2pt + rgb("#2c3e50")), width: 100%, body,
)

#align(center)[
  #text(size: 17pt, weight: "bold")[Strictness and validity push-down]
  #v(-0.4em)
  #text(size: 12pt)[the same value law, once partiality is accounted for]
]

#v(1em)

#lead[
  *Summary.* A row-local function may be pushed through an input's validity exactly when it is strict
  in that argument *and* remains defined after validity masks that argument. The first condition is the
  usual null-propagation meaning of `is_strict`; the second matters only for partial functions. It is
  automatic for an infallible function. Return-dtype representability, totality, speculative errors,
  and `Dense` safety remain separate concerns.
]

= Model

Scalar functions are *row-local*: output row $i$ depends only on input rows $i$. They are also assumed
deterministic and insensitive to the bytes behind nulls. Equality below is therefore *logical equality*
$eq.triple$: equal length, equal validity, and equal values at valid rows.

A mask is a non-nullable boolean column. It applies validity without changing valid values:

$ mask(a, m)[i] = cases(#N &"if" not m[i], a[i] &"otherwise") $

For example, masking does not distinguish a newly nulled row from one that was already null:

#figure(
  table(
    columns: 4,
    align: center,
    table.header([$i$], [$a$], [$m$], [$mask(a, m)$]),
    [0], [10], [`true`], [10],
    [1], [20], [`false`], N,
    [2], N, [true], N,
  ),
  caption: [Rows 1 and 2 are both null after masking, for different reasons.],
)

The function $f$ may be partial: an evaluation can error instead of returning a column. Statements
about its result are quantified only where that evaluation succeeds.

= The law and its missing premise

Fix an argument position $j$.

#lead[
  *$(S_j)$ Strictness.* If $f(a_1, ..., a_k)$ succeeds and $a_j[i] = #N$, its output at $i$ is #N.

  *$(C_j)$ Mask closure.* If $f(a_1, ..., a_k)$ succeeds, then
  $f(a_1, ..., mask(a_j, m), ..., a_k)$ succeeds for every mask $m$.

  *$(M_j)$ Validity equivariance.* Whenever $f(a_1, ..., a_k)$ succeeds, the masked evaluation also
  succeeds and
  $ f(a_1, ..., mask(a_j, m), ..., a_k) eq.triple mask(f(a_1, ..., a_k), m). $
]

$(M_j)$ is the law used by a validity push-down: compute after masking one argument, or compute first
and mask the result. It includes definedness of both sides, rather than treating an error as a value.

#pagebreak()

For an ordinary addition, $(M_1)$ says the following two columns agree. The evaluation after masking
is defined, and strictness makes its second row null.

#figure(
  table(
    columns: 6,
    align: center,
    table.header(
      [$i$], [$a_1$], [$a_2$], [$m$],
      [mask first, then add], [add first, then mask],
    ),
    [0], [1], [10], [`true`], [11], [11],
    [1], [2], [20], [`false`], N, N,
    [2], [3], [30], [`false`], N, N,
  ),
  caption: [The two orders differ only in the unobserved bytes behind null rows.],
)

#lead[
  *Theorem.* For a row-local deterministic function,
  $ (S_j) " and " (C_j) quad arrow.l.r quad (M_j). $
  Consequently, full strictness plus mask closure in every argument is exactly what licenses every
  per-argument validity push-down.
]

== Forward: strictness and closure imply the law

Assume $(S_j)$ and $(C_j)$, and start with any successful evaluation
$f(a_1, ..., a_k)$. By closure, the left side below also succeeds. Fix a row $i$; row-locality means
there are only two cases to check:

#figure(
  table(
    columns: (auto, 1fr, 1fr),
    align: (center, left, left),
    table.header([mask bit], [left: compute after masking], [right: mask after computing]),
    [$m[i] = $ `true`],
    [the input at row $i$ is unchanged, so this is $f(a_1, ..., a_k)[i]$],
    [masking preserves $f(a_1, ..., a_k)[i]$],
    [$m[i] = $ `false`],
    [argument $j$ is #N; the successful left evaluation is #N by $(S_j)$],
    [the mask makes the result #N by definition],
  ),
  caption: [Each row agrees, so the columns are logically equal.],
)

This proves $(M_j)$. Notice the distinct jobs of the two premises: closure establishes that the left
evaluation exists; strictness establishes its value at masked rows.

== Reverse (by contrapositive): the law implies strictness and closure

$(M_j)$ explicitly includes $(C_j)$. To obtain $(S_j)$, use its contrapositive: suppose a successful
input $b$ has a null in argument $j$ at row $i$, but gives a non-null result $v$ there. This is exactly
the negation of $(S_j)$, and we will derive a contradiction with $(M_j)$.

Choose a mask $m$ that is false only at $i$, and write
$b'_j = mask(b_j, m)$. At row $i$, $b_j[i]$ was already #N; at every other row, $m$ is true. Thus
$b'_j eq.triple b_j$. Replacing $b_j$ by $b'_j$ changes no logical input value, including at the one
row we care about.

Now apply $(M_j)$ to the successful input $b$. Its left-hand side is precisely the evaluation with
$b'_j = mask(b_j, m)$, and it guarantees that evaluation succeeds. At row $i$, the common left-hand
side has these two incompatible values:

$ f(b_1, ..., mask(b_j, m), ..., b_k)[i]
   = f(b_1, ..., b'_j, ..., b_k)[i]
   = f(b_1, ..., b_j, ..., b_k)[i] = v != #N. $

But $(M_j)$ also says

$ f(b_1, ..., mask(b_j, m), ..., b_k)[i]
   = mask(f(b_1, ..., b_j, ..., b_k), m)[i] = #N. $

The first line uses the definition of $b'_j$, then row-locality and $b'_j eq.triple b_j$; the second is
$(M_j)$ and $m[i] = $ `false`. We do not use $(S_j)$ here --- it is the fact being proved. One successful
evaluation cannot be both $v$ and #N, so the assumed counterexample cannot exist. Therefore $(M_j)$
implies $(S_j)$. $square.stroked$

#pagebreak()

The closure premise is necessary. A binary function that succeeds on $(0, 1)$, errors on $(#N, 1)$,
and otherwise returns null whenever it does evaluate with a null first argument satisfies $(S_1)$ under
the partiality convention, but not $(M_1)$: masking the first input turns a successful evaluation into
an error. Defining strictness to require a *successful* null result on every null input is an equivalent
way to build this premise into $(S_j)$.

#figure(
  table(
    columns: 4,
    align: center,
    table.header([input], [$f$], [after masking argument 1], [$f$ after masking]),
    [$(0, 1)$], [0], [$(#N, 1)$], [*error*],
  ),
  caption: [The function is vacuously strict at $(#N, 1)$ because it does not return a non-null value;
  nevertheless, it cannot satisfy the masked-evaluation law.],
)

= What the optimizer uses

The dictionary rule has the shape

#align(center)[
  #grid(
    columns: 3, column-gutter: 1.2em, align: horizon,
    node[`f(dict(codes, values), c)`],
    text(size: 13pt)[$arrow.r.long$],
    node(fill: rgb("#eafaf1"))[`dict(codes, f(values, c))`],
  )
]

A null code masks only the dictionary argument while $c$ stays live, so this requires $(M_j)$ for that
argument, not a weaker law that masks all arguments together. Kleene `AND` illustrates the difference:
`false AND NULL` is `false`, so masking only its second argument is not equivariant.

#table(
  columns: 6,
  align: center,
  table.header(
    [$a_1$], [$a_2$], [$m$], [mask $a_2$, then `AND`], [`AND`, then mask], [result],
  ),
  [`false`], [`true`], [`false`], [`false`], N, [not $(M_2)$],
)

Value equivalence is not enough for this rewrite when $f$ is fallible. It evaluates *every* dictionary
value, including values with no live code; `div(100, 0)` can then error on the rewritten side although
the original never evaluated it. Thus the dictionary rule also needs its existing no-speculative-error
condition (normally `!is_fallible`). Mask closure addresses masked input rows; it does not make dead
dictionary values safe to evaluate.

= Independent obligations

#table(
  columns: (auto, 1fr, 1fr),
  align: (left, left, left),
  table.header([property], [statement], [what it enables]),
  [strict + mask-closed], [null inputs produce null outputs and remain evaluable],
    [validity push-down],
  [representable], [the declared return dtype admits required nulls],
    [advertising `is_strict`],
  [total], [valid inputs never produce null],
    [precomputing output validity],
  [infallible], [no legal evaluation errors],
    [speculative evaluation],
  [dense-safe], [bytes behind nulls may be read safely],
    [`RowPolicy::Dense`],
)

Representability is a type-level obligation: a strict `cast` with a pinned non-nullable return type
cannot represent the null its value semantics demand. Totality is different again. A strict `list_sum`
may return null for a valid empty list, so strictness only gives

$ valid(f(a_1, ..., a_k)) subset.eq valid(a_1) " and " dots " and " valid(a_k). $

Equality, and hence a precomputed output-validity mask, additionally needs totality.

`RowFn` supplies strictness structurally. Its `Filter` path evaluates only rows valid in every input and
scatters nulls back; its `Dense` path evaluates all rows then applies that combined validity. The latter
still needs `InputElement::DENSE_SAFE`, because an invalid string view may hold unsafe bytes. That is an
operational property of an element representation, not a consequence of strictness.
