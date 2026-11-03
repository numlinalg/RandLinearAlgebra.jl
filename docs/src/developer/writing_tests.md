# Writing Tests

A checklist for adding a test file. See [Testing Overview](testing_overview.md) for the
layout and [Test Mocks](test_mocks.md) for the shared stand-in types.

## Steps

1. Put the file in the directory matching its source file, for example
   `test/Compressors/gaussian.jl` for `src/Compressors/gaussian.jl`. If the directory is new,
   add it to the list in `runtests.jl`.
2. Start from the module template:

   ```julia
   module my_test_file
   include("../test_helpers/preamble.jl")

   @testset "What is tested" begin
       # tests
   end

   end
   ```

   Adjust the relative path to the preamble for the file's depth.
3. For a recipe type, add a field check with the macro for its category, such as
   `@test_compressor`. See [Test Preamble and Helpers](test_preamble_helpers.md).
4. Use `ATOL` for floating-point comparisons.
5. Run the file on its own, then the whole suite.

## Choosing a stand-in type

- If a shared mock fits, use it with `using .MockCompressors` and pick the lowest tier that
  works.
- If the type's shape is unique to one file, define it locally in that file.
- Do not define methods on a shared mock from a test file to make it fit. Use a different
  tier, or split the file.

## Tests that expect an error before a method exists

Some tests check that a fallback throws `ArgumentError` before any method has been defined
for a type, and then define the method and check the working behavior. Such a type cannot be
shared: a method defined for it in one place would silently remove the error in another.
Keep these types local, or put the "throws" checks in their own file that uses an inert mock.
