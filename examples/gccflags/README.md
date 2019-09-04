The original gccflags.py has been modified to target code size on a
cross-compiled target and take advantage of the Combined Elimination search
technique.

The following examples show how to use Combined Elimination to tune the
selection of GCC compiler flags for code size for a cross-compiled target.

Finding working flags and parameters
-------------------------------------

The first time that ./gccflags.py is run, GCC will be queried to find the set of
available flags and parameters and whether they appear to be working. Once this
has been done the working flags and parameters are cached for future use.

Note: if you tune another target, the set of available flags and parameters may
differ. To re-query the available options add the `--no-cached-flags` option,
but note this will overwrite the previously cached flags in:
```
cc_flags.json
cc_param_defaults.json
cc_params.json
```

Tuning size for a cross-compiled target
---------------------------------------
```
  ./gccflags.py program.c \
    --output program.exe \
    --technique CombinedElimination \
    --cc /path/to/gcc-binary \
    --time-template '' \
    --objective size
```
Note: `--time-template ''` disables execution time measurement as it is not
currently supported for cross-compiled targets.

Supplying custom compile commands
---------------------------------
If your compile command is something other than `{cc} {source} -o {output} {flags}`
you'll need to supply a `--compile-template`. Here is a simple example using
make:
```

  ./gccflags.py \
    --compile-template 'make clean; make CFLAGS="{flags}"'
    --output program.exe \
    --technique CombinedElimination \
    --cc /path/to/gcc/binary \
    --time-template '' \
    --objective size

```

Supplying a specific version of the `size` utility
--------------------------------------------------
You might want to specify which version of `size` to use e.g. you might want to
use a version that supports `--format=gnu` so that rodata is excluded from text
size):
```
--size-template '/path/to/size --format=gnu {output}'
```
