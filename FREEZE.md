# FREEZE.md

Everything before the freeze commit is protocol formation and repair;
everything after it is execution against frozen rules.

## The chain

Stream D closure commits, after which no source, test, evaluation,
evidence-generation, or analysis code changed before the freeze:

    2acbfbde356b70b7094d7ec8284fbb4c070fa254  corrected verdict, dual-path
                                              disclosure, preflight table
    4787937c3b0c655c1a114cb5cc9bd7e4898e840d  ignore SLURM job logs
    72115d7b8cab19bcfaa4d09a805a86a357c803c3  evidence from the corrected run

Normative freeze commit:

    d4ed1017bd2daca2871da28900b5b4a6a7ff92b6

## The frozen artifacts

sha256 of each normative file as committed at the freeze commit. These are
hashes of the file content as git stores it, which is the content any clone
receives regardless of the checkout platform's line endings.

    517cc4924f8b770c2d27c9f9fecb761c634a920fd21a77ab44e92fe28473eee4  PROTOCOL.md
    bddd31e9d294778f10848028e4755ba56e6ea58f1e633b57eb4d09b29bbb4493  AMENDMENTS.md
    91f3a82ff066bcb029b9df7b9ff9c3da4bba31ea9604d1d17cea48a660b440e6  configs/analysis.yaml
    6e4897ff98b2061a4e8c544dfd30126e8b0b5cb561d4b8e306400f210a7c452e  VALIDATION.md

Reproduce with:

    for f in PROTOCOL.md AMENDMENTS.md configs/analysis.yaml VALIDATION.md; do
        printf '%s  %s
' "$(git show d4ed1017bd2daca2871da28900b5b4a6a7ff92b6:$f | sha256sum | cut -d' ' -f1)" "$f"
    done

## Clean-tree assertion

At the freeze commit the working tree was clean: no tracked modifications
beyond the four files staged into it, and no untracked files. The assertion
was made before the commit and re-made after it.

configs/analysis.yaml was staged unchanged. It reached its final state in an
earlier commit and is frozen here by its content hash rather than by having
been edited in this one.

## What the validator does with this

The validator's first action is to verify the four files against the hashes
above. A mismatch is a blocker before any other work begins. The freeze commit
is the normative boundary for every later judgment: a passage in PROTOCOL.md
marked "Adopted before the freeze commit" is protocol text settled during
formation, not an amendment, and AMENDMENTS.md holds only what changes after
this point. An empty AMENDMENTS.md therefore means the protocol stands exactly
as frozen.

The bookkeeping commit is the one containing this file.
