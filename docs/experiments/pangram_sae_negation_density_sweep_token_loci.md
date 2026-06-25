# Pangram SAE Token Loci

Top token activations for selected SAE nodes on explicit target rows.

## Interpretation

The dose sweep shows a token-level transition rather than just a mass increase:

- At `0` and `2` contractions, `L19:913` mostly fires on expanded auxiliary words such as `did` and `do` in `did not` / `do not` contexts.
- At `16` and `32` contractions, `L19:913` mostly fires on contraction subword pieces such as `didn`, `wouldn`, `couldn`, `wasn`, and `shouldn`.
- `L24:1310` follows the same transition but also keeps sentence-boundary activations at low contraction counts. This explains the residual baseline in the expanded text.

Class fractions from `artifacts/pangram_llama_sae/circuit_discovery/negation_density_sweep_token_loci_class_summary.csv`:

| Node | Contractions | Contraction-piece fraction | Expanded-negation fraction | Sentence-boundary fraction |
|---|---:|---:|---:|---:|
| `L19:913` | 0 | `0.00` | `0.61` | `0.11` |
| `L19:913` | 8 | `0.33` | `0.27` | `0.13` |
| `L19:913` | 16 | `0.75` | `0.00` | `0.06` |
| `L19:913` | 32 | `0.89` | `0.00` | `0.04` |
| `L24:1310` | 0 | `0.00` | `0.38` | `0.31` |
| `L24:1310` | 8 | `0.38` | `0.12` | `0.25` |
| `L24:1310` | 16 | `0.63` | `0.00` | `0.21` |
| `L24:1310` | 32 | `0.79` | `0.00` | `0.10` |

This supports a two-component interpretation: the path collects a broader negation/narrative structure at low contraction density, then shifts into a direct contraction-piece circuit as contractions become dense.

## Summary

| Node | Doc | Source | Active tokens | Total mass | Max activation | Class counts |
|---|---|---|---:|---:|---:|---|
| `L19:913` | `negation_density_sweep_contract_00` | `negation_density_sweep` | 18 | `30.69` | `5.55` | expanded_negation_word:11|punctuation:3|sentence_boundary:2|word:2 |
| `L19:913` | `negation_density_sweep_contract_02` | `negation_density_sweep` | 17 | `27.85` | `5.61` | expanded_negation_word:11|sentence_boundary:2|punctuation:2|word:2 |
| `L19:913` | `negation_density_sweep_contract_04` | `negation_density_sweep` | 16 | `26.30` | `5.53` | expanded_negation_word:8|contraction_piece:2|word:2|punctuation:2|sentence_boundary:2 |
| `L19:913` | `negation_density_sweep_contract_08` | `negation_density_sweep` | 15 | `25.96` | `5.35` | contraction_piece:5|expanded_negation_word:4|word:2|punctuation:2|sentence_boundary:2 |
| `L19:913` | `negation_density_sweep_contract_16` | `negation_density_sweep` | 16 | `39.14` | `6.07` | contraction_piece:12|punctuation:2|word:1|sentence_boundary:1 |
| `L19:913` | `negation_density_sweep_contract_32` | `negation_density_sweep` | 28 | `60.02` | `6.21` | contraction_piece:25|punctuation:1|word:1|sentence_boundary:1 |
| `L24:1310` | `negation_density_sweep_contract_00` | `negation_density_sweep` | 16 | `32.41` | `6.61` | expanded_negation_word:6|sentence_boundary:5|punctuation:3|word:2 |
| `L24:1310` | `negation_density_sweep_contract_02` | `negation_density_sweep` | 14 | `29.29` | `6.90` | expanded_negation_word:5|sentence_boundary:5|word:2|punctuation:2 |
| `L24:1310` | `negation_density_sweep_contract_04` | `negation_density_sweep` | 14 | `29.28` | `7.01` | expanded_negation_word:4|sentence_boundary:4|contraction_piece:2|word:2|punctuation:2 |
| `L24:1310` | `negation_density_sweep_contract_08` | `negation_density_sweep` | 16 | `31.26` | `5.79` | contraction_piece:6|sentence_boundary:4|expanded_negation_word:2|punctuation:2|word:2 |
| `L24:1310` | `negation_density_sweep_contract_16` | `negation_density_sweep` | 19 | `52.78` | `7.42` | contraction_piece:12|sentence_boundary:4|punctuation:2|word:1 |
| `L24:1310` | `negation_density_sweep_contract_32` | `negation_density_sweep` | 29 | `78.91` | `7.93` | contraction_piece:23|sentence_boundary:3|punctuation:2|word:1 |

## Top Tokens

- `L19:913` `negation_density_sweep_contract_00` token `462` activation `5.555` class `expanded_negation_word` token ` did`
  - every failure point. The director was not pleased, but he did not argue when the second cue failed exactly where Priya expected
- `L19:913` `negation_density_sweep_contract_00` token `319` activation `5.472` class `expanded_negation_word` token ` did`
  - flood again. He had not prepared a speech, so he did not sound polished. Still, he would not apologize for asking
- `L19:913` `negation_density_sweep_contract_00` token `285` activation `3.515` class `expanded_negation_word` token ` do`
  - was not clear about who would repair the cracked sidewalk. Residents do not usually read every appendix, he thought, but they should
- `L19:913` `negation_density_sweep_contract_00` token `260` activation `2.942` class `expanded_negation_word` token ` did`
  - let the drainage plan pass without questions. The engineer's map did not show the alley behind Cedar Street, and the cost estimate
- `L19:913` `negation_density_sweep_contract_00` token `168` activation `1.895` class `expanded_negation_word` token ` did`
  - the afternoon meeting easier. When Omar said the missing boxes probably did not matter, Lena did not answer right away. She walked
- `L19:913` `negation_density_sweep_contract_00` token `508` activation `1.879` class `expanded_negation_word` token ` did`
  - was not working. That was more useful than pretending the problems did not exist.
- `L19:913` `negation_density_sweep_contract_00` token `344` activation `1.571` class `expanded_negation_word` token ` did`
  - basic questions. When the chair tried to move on, Marcus did not sit down. He pointed to the blank corner of the
- `L19:913` `negation_density_sweep_contract_00` token `437` activation `0.959` class `expanded_negation_word` token ` did`
  - make the curtain rise any faster. She told the crew they did not need a perfect run; they needed one honest pass through
- `L19:913` `negation_density_sweep_contract_00` token `232` activation `0.904` class `expanded_negation_word` token ` did`
  - , and two customers had not received anything at all. Marcus did not want the council hearing to turn into a lecture, but
- `L19:913` `negation_density_sweep_contract_00` token `417` activation `0.836` class `expanded_negation_word` token ` did`
  - learned the scene change that everyone else had practiced. Priya did not panic, because panic would not make the curtain rise any
- `L19:913` `negation_density_sweep_contract_00` token `378` activation `0.821` class `expanded_negation_word` token ` did`
  - the people living there could see themselves in it. The rehearsal did not begin well. The lights were not focused, the stage
- `L19:913` `negation_density_sweep_contract_00` token `199` activation `0.783` class `sentence_boundary` token `.`
  - blue tag, and wrote down the numbers that did not fit. By sunset, the mistake was not dramatic, but it was
- `L19:913` `negation_density_sweep_contract_00` token `443` activation `0.670` class `punctuation` token `;`
  - . She told the crew they did not need a perfect run; they needed one honest pass through every failure point. The director
- `L19:913` `negation_density_sweep_contract_00` token `376` activation `0.667` class `word` token `The`
  - finished until the people living there could see themselves in it. The rehearsal did not begin well. The lights were not focused,
- `L19:913` `negation_density_sweep_contract_00` token `291` activation `0.622` class `punctuation` token `,`
  - repair the cracked sidewalk. Residents do not usually read every appendix, he thought, but they should not have to guess whether their
- `L19:913` `negation_density_sweep_contract_00` token `472` activation `0.603` class `word` token ` Pri`
  - but he did not argue when the second cue failed exactly where Priya expected. By the end of the night, the show
- `L19:913` `negation_density_sweep_contract_02` token `462` activation `5.605` class `expanded_negation_word` token ` did`
  - every failure point. The director was not pleased, but he did not argue when the second cue failed exactly where Priya expected
- `L19:913` `negation_density_sweep_contract_02` token `319` activation `5.418` class `expanded_negation_word` token ` did`
  - flood again. He had not prepared a speech, so he did not sound polished. Still, he would not apologize for asking
- `L19:913` `negation_density_sweep_contract_02` token `285` activation `3.043` class `expanded_negation_word` token ` do`
  - was not clear about who would repair the cracked sidewalk. Residents do not usually read every appendix, he thought, but they should
- `L19:913` `negation_density_sweep_contract_02` token `260` activation `2.625` class `expanded_negation_word` token ` did`
  - let the drainage plan pass without questions. The engineer's map did not show the alley behind Cedar Street, and the cost estimate
- `L19:913` `negation_density_sweep_contract_02` token `508` activation `1.944` class `expanded_negation_word` token ` did`
  - was not working. That was more useful than pretending the problems did not exist.
- `L19:913` `negation_density_sweep_contract_02` token `344` activation `1.564` class `expanded_negation_word` token ` did`
  - basic questions. When the chair tried to move on, Marcus did not sit down. He pointed to the blank corner of the
- `L19:913` `negation_density_sweep_contract_02` token `437` activation `0.938` class `expanded_negation_word` token ` did`
  - make the curtain rise any faster. She told the crew they did not need a perfect run; they needed one honest pass through
- `L19:913` `negation_density_sweep_contract_02` token `168` activation `0.908` class `expanded_negation_word` token ` did`
  - the afternoon meeting easier. When Omar said the missing boxes probably did not matter, Lena did not answer right away. She walked
- `L19:913` `negation_density_sweep_contract_02` token `199` activation `0.851` class `sentence_boundary` token `.`
  - blue tag, and wrote down the numbers that did not fit. By sunset, the mistake was not dramatic, but it was
- `L19:913` `negation_density_sweep_contract_02` token `232` activation `0.687` class `expanded_negation_word` token ` did`
  - , and two customers had not received anything at all. Marcus did not want the council hearing to turn into a lecture, but
- `L19:913` `negation_density_sweep_contract_02` token `378` activation `0.683` class `expanded_negation_word` token ` did`
  - the people living there could see themselves in it. The rehearsal did not begin well. The lights were not focused, the stage
- `L19:913` `negation_density_sweep_contract_02` token `443` activation `0.675` class `punctuation` token `;`
  - . She told the crew they did not need a perfect run; they needed one honest pass through every failure point. The director
- `L19:913` `negation_density_sweep_contract_02` token `376` activation `0.665` class `word` token `The`
  - finished until the people living there could see themselves in it. The rehearsal did not begin well. The lights were not focused,
- `L19:913` `negation_density_sweep_contract_02` token `472` activation `0.608` class `word` token ` Pri`
  - but he did not argue when the second cue failed exactly where Priya expected. By the end of the night, the show
- `L19:913` `negation_density_sweep_contract_02` token `291` activation `0.582` class `punctuation` token `,`
  - repair the cracked sidewalk. Residents do not usually read every appendix, he thought, but they should not have to guess whether their
- `L19:913` `negation_density_sweep_contract_02` token `375` activation `0.529` class `sentence_boundary` token `.

`
  - not finished until the people living there could see themselves in it. The rehearsal did not begin well. The lights were not focused
- `L19:913` `negation_density_sweep_contract_04` token `462` activation `5.527` class `expanded_negation_word` token ` did`
  - every failure point. The director was not pleased, but he did not argue when the second cue failed exactly where Priya expected
- `L19:913` `negation_density_sweep_contract_04` token `319` activation `5.375` class `expanded_negation_word` token ` did`
  - flood again. He had not prepared a speech, so he did not sound polished. Still, he would not apologize for asking
- `L19:913` `negation_density_sweep_contract_04` token `168` activation `3.104` class `contraction_piece` token ` didn`
  - the afternoon meeting easier. When Omar said the missing boxes probably didn't matter, Lena did not answer right away. She walked
- `L19:913` `negation_density_sweep_contract_04` token `285` activation `2.049` class `expanded_negation_word` token ` do`
  - was not clear about who would repair the cracked sidewalk. Residents do not usually read every appendix, he thought, but they should
- `L19:913` `negation_density_sweep_contract_04` token `508` activation `1.866` class `expanded_negation_word` token ` did`
  - was not working. That was more useful than pretending the problems did not exist.
- `L19:913` `negation_density_sweep_contract_04` token `344` activation `1.636` class `expanded_negation_word` token ` did`
  - basic questions. When the chair tried to move on, Marcus did not sit down. He pointed to the blank corner of the
- `L19:913` `negation_density_sweep_contract_04` token `99` activation `0.933` class `contraction_piece` token ` didn`
  - manager asked why she was still at her desk. The totals didn't match the pallets by the loading door, the return
- `L19:913` `negation_density_sweep_contract_04` token `437` activation `0.927` class `expanded_negation_word` token ` did`
  - make the curtain rise any faster. She told the crew they did not need a perfect run; they needed one honest pass through
- `L19:913` `negation_density_sweep_contract_04` token `376` activation `0.677` class `word` token `The`
  - finished until the people living there could see themselves in it. The rehearsal did not begin well. The lights were not focused,
- `L19:913` `negation_density_sweep_contract_04` token `443` activation `0.663` class `punctuation` token `;`
  - . She told the crew they did not need a perfect run; they needed one honest pass through every failure point. The director
- `L19:913` `negation_density_sweep_contract_04` token `472` activation `0.645` class `word` token ` Pri`
  - but he did not argue when the second cue failed exactly where Priya expected. By the end of the night, the show
- `L19:913` `negation_density_sweep_contract_04` token `378` activation `0.624` class `expanded_negation_word` token ` did`
  - the people living there could see themselves in it. The rehearsal did not begin well. The lights were not focused, the stage
- `L19:913` `negation_density_sweep_contract_04` token `199` activation `0.602` class `sentence_boundary` token `.`
  - blue tag, and wrote down the numbers that did not fit. By sunset, the mistake was not dramatic, but it was
- `L19:913` `negation_density_sweep_contract_04` token `291` activation `0.562` class `punctuation` token `,`
  - repair the cracked sidewalk. Residents do not usually read every appendix, he thought, but they should not have to guess whether their
- `L19:913` `negation_density_sweep_contract_04` token `232` activation `0.559` class `expanded_negation_word` token ` did`
  - , and two customers had not received anything at all. Marcus did not want the council hearing to turn into a lecture, but
- `L19:913` `negation_density_sweep_contract_04` token `375` activation `0.549` class `sentence_boundary` token `.

`
  - not finished until the people living there could see themselves in it. The rehearsal did not begin well. The lights were not focused
- `L19:913` `negation_density_sweep_contract_08` token `462` activation `5.348` class `expanded_negation_word` token ` did`
  - every failure point. The director was not pleased, but he did not argue when the second cue failed exactly where Priya expected
- `L19:913` `negation_density_sweep_contract_08` token `232` activation `4.602` class `contraction_piece` token ` didn`
  - , and two customers had not received anything at all. Marcus didn't want the council hearing to turn into a lecture, but
- `L19:913` `negation_density_sweep_contract_08` token `168` activation `3.104` class `contraction_piece` token ` didn`
  - the afternoon meeting easier. When Omar said the missing boxes probably didn't matter, Lena didn't answer right away. She walked
- `L19:913` `negation_density_sweep_contract_08` token `260` activation `2.310` class `contraction_piece` token ` didn`
  - let the drainage plan pass without questions. The engineer's map didn't show the alley behind Cedar Street, and the cost estimate
- `L19:913` `negation_density_sweep_contract_08` token `508` activation `1.814` class `expanded_negation_word` token ` did`
  - was not working. That was more useful than pretending the problems did not exist.
- `L19:913` `negation_density_sweep_contract_08` token `196` activation `1.798` class `contraction_piece` token ` didn`
  - , counted every blue tag, and wrote down the numbers that didn't fit. By sunset, the mistake was not dramatic,
- `L19:913` `negation_density_sweep_contract_08` token `319` activation `1.638` class `expanded_negation_word` token ` did`
  - flood again. He had not prepared a speech, so he did not sound polished. Still, he would not apologize for asking
- `L19:913` `negation_density_sweep_contract_08` token `437` activation `0.955` class `expanded_negation_word` token ` did`
  - make the curtain rise any faster. She told the crew they did not need a perfect run; they needed one honest pass through
- `L19:913` `negation_density_sweep_contract_08` token `99` activation `0.933` class `contraction_piece` token ` didn`
  - manager asked why she was still at her desk. The totals didn't match the pallets by the loading door, the return
- `L19:913` `negation_density_sweep_contract_08` token `376` activation `0.666` class `word` token `The`
  - finished until the people living there could see themselves in it. The rehearsal did not begin well. The lights were not focused,
- `L19:913` `negation_density_sweep_contract_08` token `443` activation `0.656` class `punctuation` token `;`
  - . She told the crew they did not need a perfect run; they needed one honest pass through every failure point. The director
- `L19:913` `negation_density_sweep_contract_08` token `291` activation `0.635` class `punctuation` token `,`
  - repair the cracked sidewalk. Residents do not usually read every appendix, he thought, but they should not have to guess whether their
- `L19:913` `negation_density_sweep_contract_08` token `472` activation `0.571` class `word` token ` Pri`
  - but he did not argue when the second cue failed exactly where Priya expected. By the end of the night, the show
- `L19:913` `negation_density_sweep_contract_08` token `375` activation `0.500` class `sentence_boundary` token `.

`
  - not finished until the people living there could see themselves in it. The rehearsal did not begin well. The lights were not focused
- `L19:913` `negation_density_sweep_contract_08` token `199` activation `0.430` class `sentence_boundary` token `.`
  - blue tag, and wrote down the numbers that didn't fit. By sunset, the mistake was not dramatic, but it was
- `L19:913` `negation_density_sweep_contract_16` token `437` activation `6.066` class `contraction_piece` token ` didn`
  - make the curtain rise any faster. She told the crew they didn't need a perfect run; they needed one honest pass through
- `L19:913` `negation_density_sweep_contract_16` token `319` activation `5.689` class `contraction_piece` token ` didn`
  - flood again. He had not prepared a speech, so he didn't sound polished. Still, he would not apologize for asking
- `L19:913` `negation_density_sweep_contract_16` token `232` activation `4.595` class `contraction_piece` token ` didn`
  - , and two customers had not received anything at all. Marcus didn't want the council hearing to turn into a lecture, but
- `L19:913` `negation_density_sweep_contract_16` token `508` activation `4.300` class `contraction_piece` token ` didn`
  - was not working. That was more useful than pretending the problems didn't exist.
- `L19:913` `negation_density_sweep_contract_16` token `168` activation `3.315` class `contraction_piece` token ` didn`
  - the afternoon meeting easier. When Omar said the missing boxes probably didn't matter, Lena didn't answer right away. She walked
- `L19:913` `negation_density_sweep_contract_16` token `462` activation `2.549` class `contraction_piece` token ` didn`
  - every failure point. The director was not pleased, but he didn't argue when the second cue failed exactly where Priya expected
- `L19:913` `negation_density_sweep_contract_16` token `260` activation `2.352` class `contraction_piece` token ` didn`
  - let the drainage plan pass without questions. The engineer's map didn't show the alley behind Cedar Street, and the cost estimate
- `L19:913` `negation_density_sweep_contract_16` token `196` activation `2.226` class `contraction_piece` token ` didn`
  - , counted every blue tag, and wrote down the numbers that didn't fit. By sunset, the mistake was not dramatic,
- `L19:913` `negation_density_sweep_contract_16` token `344` activation `1.879` class `contraction_piece` token ` didn`
  - basic questions. When the chair tried to move on, Marcus didn't sit down. He pointed to the blank corner of the
- `L19:913` `negation_density_sweep_contract_16` token `378` activation `1.827` class `contraction_piece` token ` didn`
  - the people living there could see themselves in it. The rehearsal didn't begin well. The lights were not focused, the stage
- `L19:913` `negation_density_sweep_contract_16` token `417` activation `1.100` class `contraction_piece` token ` didn`
  - learned the scene change that everyone else had practiced. Priya didn't panic, because panic would not make the curtain rise any
- `L19:913` `negation_density_sweep_contract_16` token `99` activation `0.933` class `contraction_piece` token ` didn`
  - manager asked why she was still at her desk. The totals didn't match the pallets by the loading door, the return
- `L19:913` `negation_density_sweep_contract_16` token `291` activation `0.648` class `punctuation` token `,`
  - repair the cracked sidewalk. Residents do not usually read every appendix, he thought, but they should not have to guess whether their
- `L19:913` `negation_density_sweep_contract_16` token `443` activation `0.621` class `punctuation` token `;`
  - . She told the crew they didn't need a perfect run; they needed one honest pass through every failure point. The director
- `L19:913` `negation_density_sweep_contract_16` token `376` activation `0.602` class `word` token `The`
  - finished until the people living there could see themselves in it. The rehearsal didn't begin well. The lights were not focused,
- `L19:913` `negation_density_sweep_contract_16` token `199` activation `0.435` class `sentence_boundary` token `.`
  - blue tag, and wrote down the numbers that didn't fit. By sunset, the mistake was not dramatic, but it was
- `L19:913` `negation_density_sweep_contract_32` token `423` activation `6.214` class `contraction_piece` token ` wouldn`
  - else had practiced. Priya didn't panic, because panic wouldn't make the curtain rise any faster. She told the crew
- `L19:913` `negation_density_sweep_contract_32` token `246` activation `5.631` class `contraction_piece` token ` couldn`
  - want the council hearing to turn into a lecture, but he couldn't let the drainage plan pass without questions. The engineer's
- `L19:913` `negation_density_sweep_contract_32` token `327` activation `5.297` class `contraction_piece` token ` wouldn`
  - speech, so he didn't sound polished. Still, he wouldn't apologize for asking basic questions. When the chair tried to
- `L19:913` `negation_density_sweep_contract_32` token `205` activation `4.426` class `contraction_piece` token ` wasn`
  - the numbers that didn't fit. By sunset, the mistake wasn't dramatic, but it was real: three orders had been
- `L19:913` `negation_density_sweep_contract_32` token `173` activation `3.950` class `contraction_piece` token ` didn`
  - When Omar said the missing boxes probably didn't matter, Lena didn't answer right away. She walked the aisle again, counted
- `L19:913` `negation_density_sweep_contract_32` token `297` activation `3.578` class `contraction_piece` token ` shouldn`
  - don't usually read every appendix, he thought, but they shouldn't have to guess whether their basements would flood again.
- `L19:913` `negation_density_sweep_contract_32` token `232` activation `3.344` class `contraction_piece` token ` didn`
  - , and two customers had not received anything at all. Marcus didn't want the council hearing to turn into a lecture, but
- `L19:913` `negation_density_sweep_contract_32` token `462` activation `2.742` class `contraction_piece` token ` didn`
  - every failure point. The director wasn't pleased, but he didn't argue when the second cue failed exactly where Priya expected
- `L19:913` `negation_density_sweep_contract_32` token `148` activation `2.424` class `contraction_piece` token ` wouldn`
  - a bad scanner could explain part of the problem, but she wouldn't close the file just to make the afternoon meeting easier.
- `L19:913` `negation_density_sweep_contract_32` token `486` activation `2.325` class `contraction_piece` token ` wasn`
  - expected. By the end of the night, the show still wasn't ready, yet the team finally knew what wasn't working
- `L19:913` `negation_density_sweep_contract_32` token `273` activation `1.824` class `contraction_piece` token ` wasn`
  - 't show the alley behind Cedar Street, and the cost estimate wasn't clear about who would repair the cracked sidewalk. Residents don
- `L19:913` `negation_density_sweep_contract_32` token `285` activation `1.726` class `contraction_piece` token ` don`
  - wasn't clear about who would repair the cracked sidewalk. Residents don't usually read every appendix, he thought, but they shouldn
- `L19:913` `negation_density_sweep_contract_32` token `196` activation `1.681` class `contraction_piece` token ` didn`
  - , counted every blue tag, and wrote down the numbers that didn't fit. By sunset, the mistake wasn't dramatic,
- `L19:913` `negation_density_sweep_contract_32` token `392` activation `1.678` class `contraction_piece` token ` couldn`
  - begin well. The lights weren't focused, the stage manager couldn't find the spare headset, and the lead actor had not
- `L19:913` `negation_density_sweep_contract_32` token `129` activation `1.655` class `contraction_piece` token ` couldn`
  - and the shipment log wasn't signed by the driver. She couldn't accuse anyone yet, because a bad scanner could explain part
- `L19:913` `negation_density_sweep_contract_32` token `508` activation `1.631` class `contraction_piece` token ` didn`
  - wasn't working. That was more useful than pretending the problems didn't exist.
- `L24:1310` `negation_density_sweep_contract_00` token `319` activation `6.614` class `expanded_negation_word` token ` did`
  - flood again. He had not prepared a speech, so he did not sound polished. Still, he would not apologize for asking
- `L24:1310` `negation_density_sweep_contract_00` token `462` activation `5.889` class `expanded_negation_word` token ` did`
  - every failure point. The director was not pleased, but he did not argue when the second cue failed exactly where Priya expected
- `L24:1310` `negation_density_sweep_contract_00` token `285` activation `4.196` class `expanded_negation_word` token ` do`
  - was not clear about who would repair the cracked sidewalk. Residents do not usually read every appendix, he thought, but they should
- `L24:1310` `negation_density_sweep_contract_00` token `260` activation `3.237` class `expanded_negation_word` token ` did`
  - let the drainage plan pass without questions. The engineer's map did not show the alley behind Cedar Street, and the cost estimate
- `L24:1310` `negation_density_sweep_contract_00` token `168` activation `2.441` class `expanded_negation_word` token ` did`
  - the afternoon meeting easier. When Omar said the missing boxes probably did not matter, Lena did not answer right away. She walked
- `L24:1310` `negation_density_sweep_contract_00` token `199` activation `1.519` class `sentence_boundary` token `.`
  - blue tag, and wrote down the numbers that did not fit. By sunset, the mistake was not dramatic, but it was
- `L24:1310` `negation_density_sweep_contract_00` token `160` activation `1.320` class `sentence_boundary` token `.`
  - would not close the file just to make the afternoon meeting easier. When Omar said the missing boxes probably did not matter, Lena
- `L24:1310` `negation_density_sweep_contract_00` token `376` activation `0.934` class `word` token `The`
  - finished until the people living there could see themselves in it. The rehearsal did not begin well. The lights were not focused,
- `L24:1310` `negation_density_sweep_contract_00` token `243` activation `0.928` class `punctuation` token `,`
  - Marcus did not want the council hearing to turn into a lecture, but he could not let the drainage plan pass without questions.
- `L24:1310` `negation_density_sweep_contract_00` token `375` activation `0.896` class `sentence_boundary` token `.

`
  - not finished until the people living there could see themselves in it. The rehearsal did not begin well. The lights were not focused
- `L24:1310` `negation_density_sweep_contract_00` token `472` activation `0.826` class `word` token ` Pri`
  - but he did not argue when the second cue failed exactly where Priya expected. By the end of the night, the show
- `L24:1310` `negation_density_sweep_contract_00` token `382` activation `0.821` class `sentence_boundary` token `.`
  - could see themselves in it. The rehearsal did not begin well. The lights were not focused, the stage manager could not find
- `L24:1310` `negation_density_sweep_contract_00` token `489` activation `0.751` class `punctuation` token `,`
  - the end of the night, the show still was not ready, yet the team finally knew what was not working. That was
- `L24:1310` `negation_density_sweep_contract_00` token `291` activation `0.701` class `punctuation` token `,`
  - repair the cracked sidewalk. Residents do not usually read every appendix, he thought, but they should not have to guess whether their
- `L24:1310` `negation_density_sweep_contract_00` token `344` activation `0.696` class `expanded_negation_word` token ` did`
  - basic questions. When the chair tried to move on, Marcus did not sit down. He pointed to the blank corner of the
- `L24:1310` `negation_density_sweep_contract_00` token `309` activation `0.643` class `sentence_boundary` token `.`
  - should not have to guess whether their basements would flood again. He had not prepared a speech, so he did not sound
- `L24:1310` `negation_density_sweep_contract_02` token `319` activation `6.897` class `expanded_negation_word` token ` did`
  - flood again. He had not prepared a speech, so he did not sound polished. Still, he would not apologize for asking
- `L24:1310` `negation_density_sweep_contract_02` token `462` activation `6.223` class `expanded_negation_word` token ` did`
  - every failure point. The director was not pleased, but he did not argue when the second cue failed exactly where Priya expected
- `L24:1310` `negation_density_sweep_contract_02` token `285` activation `3.688` class `expanded_negation_word` token ` do`
  - was not clear about who would repair the cracked sidewalk. Residents do not usually read every appendix, he thought, but they should
- `L24:1310` `negation_density_sweep_contract_02` token `260` activation `2.837` class `expanded_negation_word` token ` did`
  - let the drainage plan pass without questions. The engineer's map did not show the alley behind Cedar Street, and the cost estimate
- `L24:1310` `negation_density_sweep_contract_02` token `199` activation `1.579` class `sentence_boundary` token `.`
  - blue tag, and wrote down the numbers that did not fit. By sunset, the mistake was not dramatic, but it was
- `L24:1310` `negation_density_sweep_contract_02` token `160` activation `1.160` class `sentence_boundary` token `.`
  - would not close the file just to make the afternoon meeting easier. When Omar said the missing boxes probably did not matter, Lena
- `L24:1310` `negation_density_sweep_contract_02` token `375` activation `1.058` class `sentence_boundary` token `.

`
  - not finished until the people living there could see themselves in it. The rehearsal did not begin well. The lights were not focused
- `L24:1310` `negation_density_sweep_contract_02` token `376` activation `1.045` class `word` token `The`
  - finished until the people living there could see themselves in it. The rehearsal did not begin well. The lights were not focused,
- `L24:1310` `negation_density_sweep_contract_02` token `382` activation `0.890` class `sentence_boundary` token `.`
  - could see themselves in it. The rehearsal did not begin well. The lights were not focused, the stage manager could not find
- `L24:1310` `negation_density_sweep_contract_02` token `472` activation `0.880` class `word` token ` Pri`
  - but he did not argue when the second cue failed exactly where Priya expected. By the end of the night, the show
- `L24:1310` `negation_density_sweep_contract_02` token `243` activation `0.841` class `punctuation` token `,`
  - Marcus did not want the council hearing to turn into a lecture, but he could not let the drainage plan pass without questions.
- `L24:1310` `negation_density_sweep_contract_02` token `344` activation `0.784` class `expanded_negation_word` token ` did`
  - basic questions. When the chair tried to move on, Marcus did not sit down. He pointed to the blank corner of the
- `L24:1310` `negation_density_sweep_contract_02` token `489` activation `0.745` class `punctuation` token `,`
  - the end of the night, the show still was not ready, yet the team finally knew what was not working. That was
- `L24:1310` `negation_density_sweep_contract_02` token `309` activation `0.661` class `sentence_boundary` token `.`
  - should not have to guess whether their basements would flood again. He had not prepared a speech, so he did not sound
- `L24:1310` `negation_density_sweep_contract_04` token `319` activation `7.011` class `expanded_negation_word` token ` did`
  - flood again. He had not prepared a speech, so he did not sound polished. Still, he would not apologize for asking
- `L24:1310` `negation_density_sweep_contract_04` token `462` activation `6.065` class `expanded_negation_word` token ` did`
  - every failure point. The director was not pleased, but he did not argue when the second cue failed exactly where Priya expected
- `L24:1310` `negation_density_sweep_contract_04` token `168` activation `4.368` class `contraction_piece` token ` didn`
  - the afternoon meeting easier. When Omar said the missing boxes probably didn't matter, Lena did not answer right away. She walked
- `L24:1310` `negation_density_sweep_contract_04` token `285` activation `2.201` class `expanded_negation_word` token ` do`
  - was not clear about who would repair the cracked sidewalk. Residents do not usually read every appendix, he thought, but they should
- `L24:1310` `negation_density_sweep_contract_04` token `99` activation `1.426` class `contraction_piece` token ` didn`
  - manager asked why she was still at her desk. The totals didn't match the pallets by the loading door, the return
- `L24:1310` `negation_density_sweep_contract_04` token `375` activation `1.085` class `sentence_boundary` token `.

`
  - not finished until the people living there could see themselves in it. The rehearsal did not begin well. The lights were not focused
- `L24:1310` `negation_density_sweep_contract_04` token `199` activation `1.074` class `sentence_boundary` token `.`
  - blue tag, and wrote down the numbers that did not fit. By sunset, the mistake was not dramatic, but it was
- `L24:1310` `negation_density_sweep_contract_04` token `344` activation `1.067` class `expanded_negation_word` token ` did`
  - basic questions. When the chair tried to move on, Marcus did not sit down. He pointed to the blank corner of the
- `L24:1310` `negation_density_sweep_contract_04` token `376` activation `1.023` class `word` token `The`
  - finished until the people living there could see themselves in it. The rehearsal did not begin well. The lights were not focused,
- `L24:1310` `negation_density_sweep_contract_04` token `472` activation `0.918` class `word` token ` Pri`
  - but he did not argue when the second cue failed exactly where Priya expected. By the end of the night, the show
- `L24:1310` `negation_density_sweep_contract_04` token `382` activation `0.863` class `sentence_boundary` token `.`
  - could see themselves in it. The rehearsal did not begin well. The lights were not focused, the stage manager could not find
- `L24:1310` `negation_density_sweep_contract_04` token `243` activation `0.773` class `punctuation` token `,`
  - Marcus did not want the council hearing to turn into a lecture, but he could not let the drainage plan pass without questions.
- `L24:1310` `negation_density_sweep_contract_04` token `489` activation `0.722` class `punctuation` token `,`
  - the end of the night, the show still was not ready, yet the team finally knew what was not working. That was
- `L24:1310` `negation_density_sweep_contract_04` token `309` activation `0.682` class `sentence_boundary` token `.`
  - should not have to guess whether their basements would flood again. He had not prepared a speech, so he did not sound
- `L24:1310` `negation_density_sweep_contract_08` token `462` activation `5.786` class `expanded_negation_word` token ` did`
  - every failure point. The director was not pleased, but he did not argue when the second cue failed exactly where Priya expected
- `L24:1310` `negation_density_sweep_contract_08` token `232` activation `5.655` class `contraction_piece` token ` didn`
  - , and two customers had not received anything at all. Marcus didn't want the council hearing to turn into a lecture, but
- `L24:1310` `negation_density_sweep_contract_08` token `168` activation `4.368` class `contraction_piece` token ` didn`
  - the afternoon meeting easier. When Omar said the missing boxes probably didn't matter, Lena didn't answer right away. She walked
- `L24:1310` `negation_density_sweep_contract_08` token `260` activation `3.102` class `contraction_piece` token ` didn`
  - let the drainage plan pass without questions. The engineer's map didn't show the alley behind Cedar Street, and the cost estimate
- `L24:1310` `negation_density_sweep_contract_08` token `196` activation `2.867` class `contraction_piece` token ` didn`
  - , counted every blue tag, and wrote down the numbers that didn't fit. By sunset, the mistake was not dramatic,
- `L24:1310` `negation_density_sweep_contract_08` token `99` activation `1.426` class `contraction_piece` token ` didn`
  - manager asked why she was still at her desk. The totals didn't match the pallets by the loading door, the return
- `L24:1310` `negation_density_sweep_contract_08` token `243` activation `1.000` class `punctuation` token `,`
  - Marcus didn't want the council hearing to turn into a lecture, but he could not let the drainage plan pass without questions.
- `L24:1310` `negation_density_sweep_contract_08` token `376` activation `0.979` class `word` token `The`
  - finished until the people living there could see themselves in it. The rehearsal did not begin well. The lights were not focused,
- `L24:1310` `negation_density_sweep_contract_08` token `375` activation `0.960` class `sentence_boundary` token `.

`
  - not finished until the people living there could see themselves in it. The rehearsal did not begin well. The lights were not focused
- `L24:1310` `negation_density_sweep_contract_08` token `199` activation `0.865` class `sentence_boundary` token `.`
  - blue tag, and wrote down the numbers that didn't fit. By sunset, the mistake was not dramatic, but it was
- `L24:1310` `negation_density_sweep_contract_08` token `472` activation `0.813` class `word` token ` Pri`
  - but he did not argue when the second cue failed exactly where Priya expected. By the end of the night, the show
- `L24:1310` `negation_density_sweep_contract_08` token `319` activation `0.774` class `expanded_negation_word` token ` did`
  - flood again. He had not prepared a speech, so he did not sound polished. Still, he would not apologize for asking
- `L24:1310` `negation_density_sweep_contract_08` token `382` activation `0.764` class `sentence_boundary` token `.`
  - could see themselves in it. The rehearsal did not begin well. The lights were not focused, the stage manager could not find
- `L24:1310` `negation_density_sweep_contract_08` token `489` activation `0.704` class `punctuation` token `,`
  - the end of the night, the show still was not ready, yet the team finally knew what was not working. That was
- `L24:1310` `negation_density_sweep_contract_08` token `309` activation `0.629` class `sentence_boundary` token `.`
  - should not have to guess whether their basements would flood again. He had not prepared a speech, so he did not sound
- `L24:1310` `negation_density_sweep_contract_08` token `173` activation `0.564` class `contraction_piece` token ` didn`
  - When Omar said the missing boxes probably didn't matter, Lena didn't answer right away. She walked the aisle again, counted
- `L24:1310` `negation_density_sweep_contract_16` token `319` activation `7.419` class `contraction_piece` token ` didn`
  - flood again. He had not prepared a speech, so he didn't sound polished. Still, he would not apologize for asking
- `L24:1310` `negation_density_sweep_contract_16` token `437` activation `7.301` class `contraction_piece` token ` didn`
  - make the curtain rise any faster. She told the crew they didn't need a perfect run; they needed one honest pass through
- `L24:1310` `negation_density_sweep_contract_16` token `232` activation `5.799` class `contraction_piece` token ` didn`
  - , and two customers had not received anything at all. Marcus didn't want the council hearing to turn into a lecture, but
- `L24:1310` `negation_density_sweep_contract_16` token `508` activation `5.156` class `contraction_piece` token ` didn`
  - was not working. That was more useful than pretending the problems didn't exist.
- `L24:1310` `negation_density_sweep_contract_16` token `168` activation `4.706` class `contraction_piece` token ` didn`
  - the afternoon meeting easier. When Omar said the missing boxes probably didn't matter, Lena didn't answer right away. She walked
- `L24:1310` `negation_density_sweep_contract_16` token `196` activation `3.534` class `contraction_piece` token ` didn`
  - , counted every blue tag, and wrote down the numbers that didn't fit. By sunset, the mistake was not dramatic,
- `L24:1310` `negation_density_sweep_contract_16` token `260` activation `3.149` class `contraction_piece` token ` didn`
  - let the drainage plan pass without questions. The engineer's map didn't show the alley behind Cedar Street, and the cost estimate
- `L24:1310` `negation_density_sweep_contract_16` token `462` activation `2.875` class `contraction_piece` token ` didn`
  - every failure point. The director was not pleased, but he didn't argue when the second cue failed exactly where Priya expected
- `L24:1310` `negation_density_sweep_contract_16` token `378` activation `2.368` class `contraction_piece` token ` didn`
  - the people living there could see themselves in it. The rehearsal didn't begin well. The lights were not focused, the stage
- `L24:1310` `negation_density_sweep_contract_16` token `344` activation `2.189` class `contraction_piece` token ` didn`
  - basic questions. When the chair tried to move on, Marcus didn't sit down. He pointed to the blank corner of the
- `L24:1310` `negation_density_sweep_contract_16` token `99` activation `1.426` class `contraction_piece` token ` didn`
  - manager asked why she was still at her desk. The totals didn't match the pallets by the loading door, the return
- `L24:1310` `negation_density_sweep_contract_16` token `417` activation `1.168` class `contraction_piece` token ` didn`
  - learned the scene change that everyone else had practiced. Priya didn't panic, because panic would not make the curtain rise any
- `L24:1310` `negation_density_sweep_contract_16` token `243` activation `0.974` class `punctuation` token `,`
  - Marcus didn't want the council hearing to turn into a lecture, but he could not let the drainage plan pass without questions.
- `L24:1310` `negation_density_sweep_contract_16` token `376` activation `0.930` class `word` token `The`
  - finished until the people living there could see themselves in it. The rehearsal didn't begin well. The lights were not focused,
- `L24:1310` `negation_density_sweep_contract_16` token `375` activation `0.928` class `sentence_boundary` token `.

`
  - not finished until the people living there could see themselves in it. The rehearsal didn't begin well. The lights were not focused
- `L24:1310` `negation_density_sweep_contract_16` token `199` activation `0.823` class `sentence_boundary` token `.`
  - blue tag, and wrote down the numbers that didn't fit. By sunset, the mistake was not dramatic, but it was
- `L24:1310` `negation_density_sweep_contract_32` token `246` activation `7.928` class `contraction_piece` token ` couldn`
  - want the council hearing to turn into a lecture, but he couldn't let the drainage plan pass without questions. The engineer's
- `L24:1310` `negation_density_sweep_contract_32` token `423` activation `7.726` class `contraction_piece` token ` wouldn`
  - else had practiced. Priya didn't panic, because panic wouldn't make the curtain rise any faster. She told the crew
- `L24:1310` `negation_density_sweep_contract_32` token `327` activation `6.470` class `contraction_piece` token ` wouldn`
  - speech, so he didn't sound polished. Still, he wouldn't apologize for asking basic questions. When the chair tried to
- `L24:1310` `negation_density_sweep_contract_32` token `205` activation `5.932` class `contraction_piece` token ` wasn`
  - the numbers that didn't fit. By sunset, the mistake wasn't dramatic, but it was real: three orders had been
- `L24:1310` `negation_density_sweep_contract_32` token `173` activation `5.384` class `contraction_piece` token ` didn`
  - When Omar said the missing boxes probably didn't matter, Lena didn't answer right away. She walked the aisle again, counted
- `L24:1310` `negation_density_sweep_contract_32` token `297` activation `4.286` class `contraction_piece` token ` shouldn`
  - don't usually read every appendix, he thought, but they shouldn't have to guess whether their basements would flood again.
- `L24:1310` `negation_density_sweep_contract_32` token `148` activation `4.079` class `contraction_piece` token ` wouldn`
  - a bad scanner could explain part of the problem, but she wouldn't close the file just to make the afternoon meeting easier.
- `L24:1310` `negation_density_sweep_contract_32` token `232` activation `3.650` class `contraction_piece` token ` didn`
  - , and two customers had not received anything at all. Marcus didn't want the council hearing to turn into a lecture, but
- `L24:1310` `negation_density_sweep_contract_32` token `486` activation `2.871` class `contraction_piece` token ` wasn`
  - expected. By the end of the night, the show still wasn't ready, yet the team finally knew what wasn't working
- `L24:1310` `negation_density_sweep_contract_32` token `196` activation `2.856` class `contraction_piece` token ` didn`
  - , counted every blue tag, and wrote down the numbers that didn't fit. By sunset, the mistake wasn't dramatic,
- `L24:1310` `negation_density_sweep_contract_32` token `462` activation `2.796` class `contraction_piece` token ` didn`
  - every failure point. The director wasn't pleased, but he didn't argue when the second cue failed exactly where Priya expected
- `L24:1310` `negation_density_sweep_contract_32` token `129` activation `2.603` class `contraction_piece` token ` couldn`
  - and the shipment log wasn't signed by the driver. She couldn't accuse anyone yet, because a bad scanner could explain part
- `L24:1310` `negation_density_sweep_contract_32` token `273` activation `2.585` class `contraction_piece` token ` wasn`
  - 't show the alley behind Cedar Street, and the cost estimate wasn't clear about who would repair the cracked sidewalk. Residents don
- `L24:1310` `negation_density_sweep_contract_32` token `392` activation `2.207` class `contraction_piece` token ` couldn`
  - begin well. The lights weren't focused, the stage manager couldn't find the spare headset, and the lead actor had not
- `L24:1310` `negation_density_sweep_contract_32` token `508` activation `1.869` class `contraction_piece` token ` didn`
  - wasn't working. That was more useful than pretending the problems didn't exist.
- `L24:1310` `negation_density_sweep_contract_32` token `168` activation `1.820` class `contraction_piece` token ` didn`
  - the afternoon meeting easier. When Omar said the missing boxes probably didn't matter, Lena didn't answer right away. She walked
