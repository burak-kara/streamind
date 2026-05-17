#!/usr/bin/env bash
set -euo pipefail

DRAFT=false
CLEAN=false

for arg in "$@"; do
    case "$arg" in
        --draft) DRAFT=true ;;
        --clean) CLEAN=true ;;
        *) echo "Unknown flag: $arg" >&2; echo "Usage: $0 [--draft] [--clean]" >&2; exit 1 ;;
    esac
done

SRCDIR="$(cd "$(dirname "$0")" && pwd)"
BUILDDIR="$SRCDIR/build"
JOBNAME="main"

if $CLEAN && [ -d "$BUILDDIR" ]; then
    rm -rf "$BUILDDIR"
fi

mkdir -p "$BUILDDIR"

PDFLATEX="pdflatex -interaction=nonstopmode -output-directory=$BUILDDIR"

cd "$SRCDIR"

# pdflatex/bibtex return non-zero on plain warnings (undefined refs etc).
# Don't let that abort the build — check the PDF afterwards instead.
set +e
$PDFLATEX "$JOBNAME.tex"

if ! $DRAFT; then
    bibtex "$BUILDDIR/$JOBNAME"
    $PDFLATEX "$JOBNAME.tex"
    $PDFLATEX "$JOBNAME.tex"
fi
set -e

if [ ! -f "$BUILDDIR/$JOBNAME.pdf" ]; then
    echo "Build failed: $BUILDDIR/$JOBNAME.pdf not produced" >&2
    exit 1
fi

cp "$BUILDDIR/$JOBNAME.pdf" "$SRCDIR/$JOBNAME.pdf"
echo "Built: $SRCDIR/$JOBNAME.pdf"
