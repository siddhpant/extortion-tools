**PLEASE CHECK THE IMPORTANT NOTICE IN THE REPO ROOT'S README BEFORE USING.**

---

# Foreign schedule / tax calculator

Do you hold 1 paisa worth of anything outside India?

Congrats! Now you are burdened with making FATCA compliant reports so that the
mafia can lazily do `==` dict comparison instead of having a good engineered
system! The Black Money Act punishes non-disclosure / non-compliance with a
massive fine and a possible imprisonment.

The filing of schedule FA is very confusing, the information by ITD is not very
clear / ambiguous, with information hidden in bits & pieces which we have to
dig as a full-time job. And not to mention the idiotic requirement to have a
lot-level granularity for each type of transaction.

Since humans aren't computers, computing everything is overwhelming and just
borderline impossible.

This program attempts to translate everything to code, so that the logic is
clear and in understandable terms, and we can have automated calculation.

With this program, you can specify your transactions in a YAML file and compute
the capital gains, dividends, schedule FA tables A2 & A3, schedule FSI, and
form 67 values. CSVs are generated for ready upload whenever we can.

The input file must be `input.yaml` in the `data` dir relative to the script -
that is, create a folder named `data` here and put `input.yaml` inside it.

Read `input_example.yaml` to know how to make your own `input.yaml`.

---

## Requirements

- Git
- Python
- Internet

Just some regular things which should be available easily on any machine.

---

## Steps to run

1. Open terminal (powershell on Windows).

2. One time setup: install requisite libraries (make a venv if you want):

```bash
$ pip install pyyaml numpy yahooquery  # If you are on Linux, best to use distro packages.
```

3. Run with `python foreign_sched_values_calc.py`.

---

## Credits

- [sahilgupta/sbi-fx-ratekeeper](https://github.com/sahilgupta/sbi-fx-ratekeeper)
- [dpguthrie/yahooquery](https://github.com/dpguthrie/yahooquery)
- Yahoo finance
