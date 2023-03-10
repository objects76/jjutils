#/bin/bash

srcpath=$(realpath "${BASH_SOURCE:-$0}")
srcdir=$(dirname $srcpath)

echo script: $srcpath
echo srcdir: $srcdir

echo curdir is $(pwd)
ln -sf $srcdir/scope_fn.py .
ln -sf $srcdir/dbg.py .

ls -l scope_fn.py dbg.py
