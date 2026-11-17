# Data

Prompt/completion JSONL files derived from the **UD_Urdu-UDTB** treebank of
Universal Dependencies (data as of UD v2.14; standard split of 4,043 train,
552 dev and 535 test sentences; 14,806 test tokens).

| File | Task | Completion format |
|---|---|---|
| `udu_lemma_{train,dev,test}.jsonl` | Lemmatization (Exp. 3) | `word/lemma ...` |
| `udu_pos_{train,dev,test}.jsonl` | UPOS tagging (Exp. 2) | `word/TAG ...` |
| `udu_joint_{train,dev,test}.jsonl` | Joint (Exp. 1) | `word/TAG/lemma ...` |

Each line is `{"prompt": "...", "completion": "..."}`, with the prompts
`Lemmatize:`, `Tag:` and `Tag and lemmatize:` followed by the space-separated
sentence.  The POS files were named `umc_pos_*` during the experiments; they
contain the same UD_Urdu-UDTB sentences.

## Licence and attribution

UD_Urdu-UDTB is licensed under **CC BY-NC-SA 4.0**.  These derived files, and
the model predictions in `../results/predictions/`, are distributed under the
same licence (attribution, non-commercial use, share-alike).

Source treebank: https://universaldependencies.org/treebanks/ur_udtb/

Please cite the treebank:

- Riyaz Ahmad Bhat, Rajesh Bhatt, Annahita Farudi, Prescott Klassen, Bhuvana
  Narasimhan, Martha Palmer, Owen Rambow, Dipti Misra Sharma, Ashwini Vaidya,
  Sri Ramagurumurthy Vishnu, and Fei Xia. *The Hindi/Urdu Treebank Project.*
  In: Handbook of Linguistic Annotation, Springer.
- Martha Palmer, Rajesh Bhatt, Bhuvana Narasimhan, Owen Rambow, Dipti Misra
  Sharma, and Fei Xia. *Hindi Syntax: Annotating Dependency, Lexical
  Predicate-Argument Structure, and Phrase Structure.* ICON 2009.
