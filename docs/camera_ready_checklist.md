# Camera-ready consistency checklist

These items were found by comparing the accepted PDF with the executable artifact and
saved checkpoint configurations. Resolve them in the camera-ready source before the final
submission.

## Required repository/paper updates

- Replace the anonymous 4open.science footnote in the introduction with the final public
  GitHub repository URL.
- Add the final public repository URL to the arXiv revision and paper metadata.
- Export Figure 1 directly from the camera-ready LaTeX source (vector PDF/SVG or a
  high-resolution PNG) and replace `sys12-iclr.png`; the current bitmap has the right paper
  content but its filename and export predate the COLM camera-ready version.

## Version strings to reconcile

- Appendix O currently reports Transformers 4.44.2. The saved executable environment uses
  Transformers 4.46.3 with TRL 0.12.1; TRL 0.12.1 declares Transformers >=4.46.0. Report
  the executable combination or recreate and verify an actually compatible 4.44.2 lock.
- Section 4.3 names Llama-3.1-8B-Instruct, while the 8B checkpoint configuration and all
  8B launch paths use `meta-llama/Meta-Llama-3-8B-Instruct`. Use the exact identifier that
  produced Table 1.
- Section 4.3 names `mistralai/Mistral-7B-Instruct-v0.1`, while aligned checkpoint
  configurations record `mistralai/Mistral-7B-Instruct-v0.3`. Confirm which checkpoint
  produced Table 8 and make the text, code, and released adapter metadata agree.

## Mathematical/statistical copy checks

- In Equation (2), the entropy-variance summation index is printed as `t=1` although the
  summand is indexed by `i`; change it to `i=1`.
- Section 5.4 states `all r² > 0.9, p > 0.001`. Verify the intended significance direction;
  a significant monotonic regression would normally be reported with `p < 0.001`.
- Table 10 prints `> .001` for several McNemar p-values while Section 5.3 reports
  `p < .001`. Reconcile the inequality signs with the statistical output.
- Report the training seed, evaluation seed, deterministic decoding setting, exact model
  revisions, and whether reported values are single runs or averages. The accepted PDF does
  not make all of these explicit.

## Artifact release checks

- Publish the System 1 and System 2 LoRA adapters (not merged full checkpoints) when model
  licenses permit, and add immutable Hugging Face links plus commit hashes to the README.
- Run `python scripts/prepare_benchmarks.py --force`, preserve the generated manifest, and
  confirm that its 14 split sizes match the README.
- Run `pytest -q` and the 10-example smoke commands for endpoint and dynamic evaluation.
- Verify every value in `results/paper_results.csv` against the final typeset Tables 1, 8,
  and 9 after any last camera-ready table edits.
- Replace placeholder/future proceedings links with the final COLM anthology or OpenReview
  URL when it becomes available.
