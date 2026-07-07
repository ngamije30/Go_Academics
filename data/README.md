# Data Folder

## raw/
Current scope is one school, one term: `excella_school_a_term1.csv`
(Excella Secondary School Rwanda, anonymized to "School A"). There is no
School B and no Term 2/3 data yet.

If a new school or term is ever collected, drop it here matching the pattern
`*_term*.csv` — `data/preprocess.py` globs this pattern and concatenates
every file it finds. Columns must already match the Go Academics schema (see
processed/ below) — the real school name is the only thing preprocess.py
remaps (via `SCHOOL_CODES`).

The UCI Student Performance Dataset (`student-mat.csv`, `student-por.csv`) is
kept here for the inactive Plan B fallback — see docs/plan_b.md. Don't mix it
into real school term files.

## processed/
Cleaned, merged, anonymized data ready for the ML pipeline. Matches this
column structure (see CLAUDE.md) — Subject/Stream values vary by school and
are not fixed to a hardcoded list:

```
StudentID | School | Gender | Stream | Term | Subject | CA_Score | Exam_Score | Attendance_Pct | Final_Result
```

`preprocess.py` also writes `encodings.json` here (LabelEncoder class
mappings), which `train.py` embeds into `model_meta.json` so inference
never has to hardcode category values.

Never commit raw files containing real student names — only School A /
School B anonymized data belongs in version control.
