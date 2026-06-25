# Pangram SAE Lexical/Style Chain Queue

This queue filters the broad cross-layer candidate graph by peak-token distributions. It is intended to find lexical/style candidates rather than period or sentence-boundary accumulators.

## 1. `L16:3800 -> L19:262 -> L20:1382 -> L24:1664`

- Lexical score: `13.920`
- Edge geomean: `6.475`; edge scores `4.396|8.919|6.924`; shared prompts `9|15|12`
- Word/punctuation: min word `0.889`, max punctuation `0.111`, max period `0.037`
- Source balance: mean AI `0.779`, min AI `0.444`, mean human `0.221`
- Joined/missing docs: min joined `23`, max missing `27`
- Node ranks: `92|9|11|19`
- Peak tokens: L16:3800 don:2|by:2|such:2|each:1|.:1|lot:1|However:1|kinds:1 || L19:262 a:7|to:4|are:2|might:2|can:2|is:2|don:2|that:2 || L20:1382 a:3|is:2|that:2|to:1|feel:1|still:1|Easy:1|can:1 || L24:1664 to:5|can:4|is:2|still:1|feel:1|methods:1|feels:1|people:1
- Model counts: L16:3800 human:15|gpt-5.5:11|gemini-3.5-flash:1 || L19:262 gpt-5.5:33|human:10|glm-5.2:3|gemini-3.5-flash:2|qwen3.6-35b:2 || L20:1382 gpt-5.5:19|gemini-3.5-flash:1|qwen3.6-35b:1|glm-5.2:1|human:1 || L24:1664 gpt-5.5:20|human:2|qwen3.6-35b:1

## 2. `L16:3800 -> L19:1437 -> L20:1382 -> L24:1664`

- Lexical score: `12.420`
- Edge geomean: `7.199`; edge scores `13.814|3.901|6.924`; shared prompts `24|8|12`
- Word/punctuation: min word `0.889`, max punctuation `0.111`, max period `0.037`
- Source balance: mean AI `0.701`, min AI `0.444`, mean human `0.299`
- Joined/missing docs: min joined `23`, max missing `27`
- Node ranks: `92|38|11|19`
- Peak tokens: L16:3800 don:2|by:2|such:2|each:1|.:1|lot:1|However:1|kinds:1 || L19:1437 such:5|don:3|a:3|doesn:2|kind:2|do:2|by:2|hasn:1 || L20:1382 a:3|is:2|that:2|to:1|feel:1|still:1|Easy:1|can:1 || L24:1664 to:5|can:4|is:2|still:1|feel:1|methods:1|feels:1|people:1
- Model counts: L16:3800 human:15|gpt-5.5:11|gemini-3.5-flash:1 || L19:1437 human:25|gpt-5.5:19|gemini-3.5-flash:3|glm-5.2:2 || L20:1382 gpt-5.5:19|gemini-3.5-flash:1|qwen3.6-35b:1|glm-5.2:1|human:1 || L24:1664 gpt-5.5:20|human:2|qwen3.6-35b:1

## 3. `L16:1975 -> L19:1325 -> L20:2290 -> L24:2940`

- Lexical score: `7.822`
- Edge geomean: `4.456`; edge scores `2.410|15.454|2.376`; shared prompts `9|22|5`
- Word/punctuation: min word `0.938`, max punctuation `0.062`, max period `0.000`
- Source balance: mean AI `0.718`, min AI `0.438`, mean human `0.282`
- Joined/missing docs: min joined `16`, max missing `34`
- Node ranks: `61|34|86|4`
- Peak tokens: L16:1975 did:1|me:1|shoes:1|mark:1|opes:1|tricks:1|ener:1|reach:1 || L19:1325 light:2|mark:2|ulation:1|fountain:1|blood:1|heels:1|day:1|full:1 || L20:2290 place:2|ulation:1|blur:1|flesh:1|replaced:1|teachers:1|otion:1|drying:1 || L24:2940 the:13|The:9|a:7|A:3|my:1|its:1
- Model counts: L16:1975 human:9|gpt-5.5:6|gemini-3.5-flash:1 || L19:1325 glm-5.2:14|gpt-5.5:12|gemini-3.5-flash:9|qwen3.6-35b:9|human:6 || L20:2290 glm-5.2:11|gemini-3.5-flash:6|qwen3.6-35b:6|human:5|gpt-5.5:5 || L24:2940 human:10|glm-5.2:9|gemini-3.5-flash:8|qwen3.6-35b:5|gpt-5.5:2

## 4. `L16:1975 -> L19:1325 -> L20:3702 -> L24:2940`

- Lexical score: `7.343`
- Edge geomean: `3.056`; edge scores `2.410|2.496|4.747`; shared prompts `9|5|9`
- Word/punctuation: min word `0.938`, max punctuation `0.062`, max period `0.000`
- Source balance: mean AI `0.649`, min AI `0.438`, mean human `0.351`
- Joined/missing docs: min joined `16`, max missing `34`
- Node ranks: `61|34|3|4`
- Peak tokens: L16:1975 did:1|me:1|shoes:1|mark:1|opes:1|tricks:1|ener:1|reach:1 || L19:1325 light:2|mark:2|ulation:1|fountain:1|blood:1|heels:1|day:1|full:1 || L20:3702 The:19|the:7|A:1|a:1 || L24:2940 the:13|The:9|a:7|A:3|my:1|its:1
- Model counts: L16:1975 human:9|gpt-5.5:6|gemini-3.5-flash:1 || L19:1325 glm-5.2:14|gpt-5.5:12|gemini-3.5-flash:9|qwen3.6-35b:9|human:6 || L20:3702 human:12|glm-5.2:6|gemini-3.5-flash:5|qwen3.6-35b:3|gpt-5.5:2 || L24:2940 human:10|glm-5.2:9|gemini-3.5-flash:8|qwen3.6-35b:5|gpt-5.5:2

## 5. `L16:1975 -> L19:1325 -> L20:2290 -> L24:457`

- Lexical score: `7.189`
- Edge geomean: `5.998`; edge scores `2.410|15.454|5.794`; shared prompts `9|22|12`
- Word/punctuation: min word `0.938`, max punctuation `0.062`, max period `0.000`
- Source balance: mean AI `0.791`, min AI `0.438`, mean human `0.209`
- Joined/missing docs: min joined `16`, max missing `34`
- Node ranks: `61|34|86|107`
- Peak tokens: L16:1975 did:1|me:1|shoes:1|mark:1|opes:1|tricks:1|ener:1|reach:1 || L19:1325 light:2|mark:2|ulation:1|fountain:1|blood:1|heels:1|day:1|full:1 || L20:2290 place:2|ulation:1|blur:1|flesh:1|replaced:1|teachers:1|otion:1|drying:1 || L24:457 bright:3|grace:3|tall:2|frame:2|high:2|flight:1|wide:1|way:1
- Model counts: L16:1975 human:9|gpt-5.5:6|gemini-3.5-flash:1 || L19:1325 glm-5.2:14|gpt-5.5:12|gemini-3.5-flash:9|qwen3.6-35b:9|human:6 || L20:2290 glm-5.2:11|gemini-3.5-flash:6|qwen3.6-35b:6|human:5|gpt-5.5:5 || L24:457 qwen3.6-35b:10|gemini-3.5-flash:10|glm-5.2:7|gpt-5.5:3

## 6. `L16:1975 -> L19:1325 -> L20:2290 -> L24:43`

- Lexical score: `6.402`
- Edge geomean: `4.937`; edge scores `2.410|15.454|3.231`; shared prompts `9|22|6`
- Word/punctuation: min word `0.938`, max punctuation `0.062`, max period `0.000`
- Source balance: mean AI `0.708`, min AI `0.438`, mean human `0.292`
- Joined/missing docs: min joined `16`, max missing `34`
- Node ranks: `61|34|86|18`
- Peak tokens: L16:1975 did:1|me:1|shoes:1|mark:1|opes:1|tricks:1|ener:1|reach:1 || L19:1325 light:2|mark:2|ulation:1|fountain:1|blood:1|heels:1|day:1|full:1 || L20:2290 place:2|ulation:1|blur:1|flesh:1|replaced:1|teachers:1|otion:1|drying:1 || L24:43 way:3|be:1|breath:1|pick:1|stay:1|rust:1|yesterday:1|course:1
- Model counts: L16:1975 human:9|gpt-5.5:6|gemini-3.5-flash:1 || L19:1325 glm-5.2:14|gpt-5.5:12|gemini-3.5-flash:9|qwen3.6-35b:9|human:6 || L20:2290 glm-5.2:11|gemini-3.5-flash:6|qwen3.6-35b:6|human:5|gpt-5.5:5 || L24:43 gpt-5.5:10|human:10|gemini-3.5-flash:5|glm-5.2:3|qwen3.6-35b:2

## 7. `L16:1975 -> L19:1325 -> L20:2290 -> L24:229`

- Lexical score: `5.898`
- Edge geomean: `3.633`; edge scores `2.410|15.454|1.288`; shared prompts `9|22|3`
- Word/punctuation: min word `0.810`, max punctuation `0.190`, max period `0.000`
- Source balance: mean AI `0.768`, min AI `0.438`, mean human `0.232`
- Joined/missing docs: min joined `16`, max missing `34`
- Node ranks: `61|34|86|8`
- Peak tokens: L16:1975 did:1|me:1|shoes:1|mark:1|opes:1|tricks:1|ener:1|reach:1 || L19:1325 light:2|mark:2|ulation:1|fountain:1|blood:1|heels:1|day:1|full:1 || L20:2290 place:2|ulation:1|blur:1|flesh:1|replaced:1|teachers:1|otion:1|drying:1 || L24:229 ,:4|rics:1|are:1|focus:1|history:1|ing:1|She:1|regions:1
- Model counts: L16:1975 human:9|gpt-5.5:6|gemini-3.5-flash:1 || L19:1325 glm-5.2:14|gpt-5.5:12|gemini-3.5-flash:9|qwen3.6-35b:9|human:6 || L20:2290 glm-5.2:11|gemini-3.5-flash:6|qwen3.6-35b:6|human:5|gpt-5.5:5 || L24:229 glm-5.2:9|qwen3.6-35b:4|gpt-5.5:4|human:2|gemini-3.5-flash:2

## 8. `L16:113 -> L19:1556 -> L20:4055 -> L24:310`

- Lexical score: `5.766`
- Edge geomean: `4.995`; edge scores `2.743|9.957|4.562`; shared prompts `9|18|15`
- Word/punctuation: min word `0.957`, max punctuation `0.043`, max period `0.000`
- Source balance: mean AI `0.698`, min AI `0.565`, mean human `0.302`
- Joined/missing docs: min joined `23`, max missing `27`
- Node ranks: `22|59|54|40`
- Peak tokens: L16:113 that:2|she:2|will:1|your:1|want:1|and:1|someone:1|similar:1 || L19:1556 best:2|your:2|a:2|she:2|are:1|dedicated:1|emies:1|will:1 || L20:4055 it:4|this:3|you:2|she:2|dedicated:1|':1|one:1|Really:1 || L24:310 you:3|seem:2|it:1|migrated:1|highly:1|ute:1|any:1|your:1
- Model counts: L16:113 human:10|gpt-5.5:8|gemini-3.5-flash:5 || L19:1556 glm-5.2:22|human:11|qwen3.6-35b:9|gemini-3.5-flash:6|gpt-5.5:2 || L20:4055 glm-5.2:16|gemini-3.5-flash:6|qwen3.6-35b:6|human:5|gpt-5.5:4 || L24:310 human:15|glm-5.2:11|qwen3.6-35b:4|gemini-3.5-flash:3|gpt-5.5:3

## 9. `L16:1975 -> L19:33 -> L20:3702 -> L24:2940`

- Lexical score: `5.628`
- Edge geomean: `2.351`; edge scores `1.526|1.795|4.747`; shared prompts `6|5|9`
- Word/punctuation: min word `0.938`, max punctuation `0.062`, max period `0.000`
- Source balance: mean AI `0.659`, min AI `0.438`, mean human `0.341`
- Joined/missing docs: min joined `16`, max missing `34`
- Node ranks: `61|36|3|4`
- Peak tokens: L16:1975 did:1|me:1|shoes:1|mark:1|opes:1|tricks:1|ener:1|reach:1 || L19:33 and:22|are:3|is:2|so:2|to:2|also:1|cannot:1|lamp:1 || L20:3702 The:19|the:7|A:1|a:1 || L24:2940 the:13|The:9|a:7|A:3|my:1|its:1
- Model counts: L16:1975 human:9|gpt-5.5:6|gemini-3.5-flash:1 || L19:33 glm-5.2:14|gemini-3.5-flash:13|qwen3.6-35b:11|gpt-5.5:8|human:4 || L20:3702 human:12|glm-5.2:6|gemini-3.5-flash:5|qwen3.6-35b:3|gpt-5.5:2 || L24:2940 human:10|glm-5.2:9|gemini-3.5-flash:8|qwen3.6-35b:5|gpt-5.5:2

## 10. `L16:2177 -> L19:913 -> L20:354 -> L24:1310`

- Lexical score: `5.326`
- Edge geomean: `5.503`; edge scores `3.119|4.375|12.216`; shared prompts `10|11|25`
- Word/punctuation: min word `1.000`, max punctuation `0.000`, max period `0.000`
- Source balance: mean AI `1.000`, min AI `1.000`, mean human `0.000`
- Joined/missing docs: min joined `34`, max missing `16`
- Node ranks: `41|40|21|48`
- Peak tokens: L16:2177 wasn:4|gri:2|murm:2|ut:2|didn:2|align:1|list:1|couldn:1 || L19:913 didn:14|wasn:11|hadn:4|doesn:3|don:3|ek:2|ern:2|kill:1 || L20:354 don:9|didn:8|wasn:5|gri:2|couldn:2|ek:2|clen:1|As:1 || L24:1310 didn:15|don:6|wasn:5|murm:3|gri:2|hadn:2|clutch:2|ait:1
- Model counts: L16:2177 gemini-3.5-flash:28|gpt-5.5:6 || L19:913 qwen3.6-35b:29|gemini-3.5-flash:12|gpt-5.5:5|glm-5.2:4 || L20:354 gpt-5.5:40|glm-5.2:3|qwen3.6-35b:2 || L24:1310 gpt-5.5:24|gemini-3.5-flash:10|glm-5.2:7|qwen3.6-35b:6

## 11. `L16:1066 -> L19:1066 -> L20:590 -> L24:2940`

- Lexical score: `4.920`
- Edge geomean: `3.562`; edge scores `4.023|26.248|0.428`; shared prompts `25|29|3`
- Word/punctuation: min word `0.911`, max punctuation `0.089`, max period `0.000`
- Source balance: mean AI `0.926`, min AI `0.706`, mean human `0.074`
- Joined/missing docs: min joined `34`, max missing `16`
- Node ranks: `89|101|46|4`
- Peak tokens: L16:1066 ,:4|with:4|world:2|heart:2|a:2|who:2|was:2|darkness:1 || L19:1066 a:8|the:6|and:3|had:2|settled:2|air:2|was:1|with:1 || L20:590 a:13|the:9|with:3|and:2|spoke:2|many:1|future:1|’s:1 || L24:2940 the:13|The:9|a:7|A:3|my:1|its:1
- Model counts: L16:1066 gpt-5.5:45 || L19:1066 gpt-5.5:48|glm-5.2:2 || L20:590 gpt-5.5:47|glm-5.2:2 || L24:2940 human:10|glm-5.2:9|gemini-3.5-flash:8|qwen3.6-35b:5|gpt-5.5:2

## 12. `L16:1751 -> L19:1437 -> L20:1382 -> L24:1664`

- Lexical score: `4.641`
- Edge geomean: `2.935`; edge scores `0.936|3.901|6.924`; shared prompts `4|8|12`
- Word/punctuation: min word `0.929`, max punctuation `0.071`, max period `0.000`
- Source balance: mean AI `0.733`, min AI `0.490`, mean human `0.267`
- Joined/missing docs: min joined `14`, max missing `36`
- Node ranks: `70|38|11|19`
- Peak tokens: L16:1751 doesn:2|Don:1|a:1|Furthermore:1|N:1|,:1|-n:1|don:1 || L19:1437 such:5|don:3|a:3|doesn:2|kind:2|do:2|by:2|hasn:1 || L20:1382 a:3|is:2|that:2|to:1|feel:1|still:1|Easy:1|can:1 || L24:1664 to:5|can:4|is:2|still:1|feel:1|methods:1|feels:1|people:1
- Model counts: L16:1751 gemini-3.5-flash:8|human:6 || L19:1437 human:25|gpt-5.5:19|gemini-3.5-flash:3|glm-5.2:2 || L20:1382 gpt-5.5:19|gemini-3.5-flash:1|qwen3.6-35b:1|glm-5.2:1|human:1 || L24:1664 gpt-5.5:20|human:2|qwen3.6-35b:1

## 13. `L16:3800 -> L19:262 -> L20:2666 -> L24:2070`

- Lexical score: `4.594`
- Edge geomean: `3.554`; edge scores `4.396|1.453|7.026`; shared prompts `9|3|12`
- Word/punctuation: min word `0.829`, max punctuation `0.171`, max period `0.037`
- Source balance: mean AI `0.747`, min AI `0.444`, mean human `0.253`
- Joined/missing docs: min joined `25`, max missing `25`
- Node ranks: `92|9|42|69`
- Peak tokens: L16:3800 don:2|by:2|such:2|each:1|.:1|lot:1|However:1|kinds:1 || L19:262 a:7|to:4|are:2|might:2|can:2|is:2|don:2|that:2 || L20:2666 ,:4|is:4|are:2|was:2|and:2|a:2|It:2|will:1 || L24:2070 was:3|,:1|drops:1|pieces:1|eyes:1|up:1|trip:1|and:1
- Model counts: L16:3800 human:15|gpt-5.5:11|gemini-3.5-flash:1 || L19:262 gpt-5.5:33|human:10|glm-5.2:3|gemini-3.5-flash:2|qwen3.6-35b:2 || L20:2666 glm-5.2:20|qwen3.6-35b:6|gpt-5.5:4|gemini-3.5-flash:3|human:2 || L24:2070 glm-5.2:10|qwen3.6-35b:7|human:5|gemini-3.5-flash:2|gpt-5.5:1

## 14. `L16:4061 -> L19:1066 -> L20:590 -> L24:2940`

- Lexical score: `4.305`
- Edge geomean: `2.817`; edge scores `1.990|26.248|0.428`; shared prompts `13|29|3`
- Word/punctuation: min word `0.977`, max punctuation `0.023`, max period `0.023`
- Source balance: mean AI `0.926`, min AI `0.706`, mean human `0.074`
- Joined/missing docs: min joined `34`, max missing `16`
- Node ranks: `33|101|46|4`
- Peak tokens: L16:4061 M:5|didn:5|Ed:2|hadn:2|wasn:2|s:1|ip:1|and:1 || L19:1066 a:8|the:6|and:3|had:2|settled:2|air:2|was:1|with:1 || L20:590 a:13|the:9|with:3|and:2|spoke:2|many:1|future:1|’s:1 || L24:2940 the:13|The:9|a:7|A:3|my:1|its:1
- Model counts: L16:4061 gpt-5.5:44 || L19:1066 gpt-5.5:48|glm-5.2:2 || L20:590 gpt-5.5:47|glm-5.2:2 || L24:2940 human:10|glm-5.2:9|gemini-3.5-flash:8|qwen3.6-35b:5|gpt-5.5:2

## 15. `L16:2177 -> L19:491 -> L20:3178 -> L24:972`

- Lexical score: `4.210`
- Edge geomean: `4.514`; edge scores `1.030|27.839|3.208`; shared prompts `5|30|6`
- Word/punctuation: min word `0.971`, max punctuation `0.029`, max period `0.000`
- Source balance: mean AI `1.000`, min AI `1.000`, mean human `0.000`
- Joined/missing docs: min joined `34`, max missing `16`
- Node ranks: `41|48|110|86`
- Peak tokens: L16:2177 wasn:4|gri:2|murm:2|ut:2|didn:2|align:1|list:1|couldn:1 || L19:491 cortisol:2|density:2|focus:1|ulatory:1|gravity:1|auxiliary:1|compression:1|distribution:1 || L20:3178 synaptic:2|a:1|anchor:1|focus:1|the:1|stabilization:1|external:1|activation:1 || L24:972 it:3|):1|s:1|diplomats:1|gallery:1|strategist:1|metric:1|well:1
- Model counts: L16:2177 gemini-3.5-flash:28|gpt-5.5:6 || L19:491 gemini-3.5-flash:28|qwen3.6-35b:10|glm-5.2:6|gpt-5.5:6 || L20:3178 gemini-3.5-flash:26|qwen3.6-35b:6|glm-5.2:6|gpt-5.5:5 || L24:972 glm-5.2:14|gemini-3.5-flash:12|qwen3.6-35b:5|gpt-5.5:4

## 16. `L16:3450 -> L19:3450 -> L20:1780 -> L24:3469`

- Lexical score: `4.161`
- Edge geomean: `5.953`; edge scores `1.601|13.717|9.610`; shared prompts `6|20|19`
- Word/punctuation: min word `0.889`, max punctuation `0.111`, max period `0.111`
- Source balance: mean AI `0.829`, min AI `0.667`, mean human `0.171`
- Joined/missing docs: min joined `18`, max missing `32`
- Node ranks: `18|119|27|47`
- Peak tokens: L16:3450 such:5|you:2|align:2|-:2|Additionally:2|of:1|one:1|known:1 || L19:3450 However:20|such:7|Furthermore:3|.:2|you:2|Additionally:1|type:1|however:1 || L20:1780 However:6|such:3|.:2|Additionally:1|k:1|type:1|you:1|United:1 || L24:3469 such:7|However:4|Additionally:2|Furthermore:2|you:2|United:1|.:1|isn:1
- Model counts: L16:3450 gemini-3.5-flash:13|gpt-5.5:5|human:1 || L19:3450 qwen3.6-35b:22|glm-5.2:9|human:6|gpt-5.5:6|gemini-3.5-flash:5 || L20:1780 human:6|gpt-5.5:5|qwen3.6-35b:4|glm-5.2:2|gemini-3.5-flash:1 || L24:3469 gpt-5.5:8|qwen3.6-35b:7|glm-5.2:4|human:4

## 17. `L16:1349 -> L19:536 -> L20:748 -> L24:4086`

- Lexical score: `4.060`
- Edge geomean: `2.405`; edge scores `0.877|7.540|2.102`; shared prompts `3|13|5`
- Word/punctuation: min word `0.889`, max punctuation `0.111`, max period `0.111`
- Source balance: mean AI `0.861`, min AI `0.789`, mean human `0.139`
- Joined/missing docs: min joined `9`, max missing `41`
- Node ranks: `10|79|73|2`
- Peak tokens: L16:1349 United:5|World:1|array:1|set:1|.:1 || L19:536 This:3|this:2|escalate:2|a:2|your:2|renewed:2|main:1|holistic:1 || L20:748 professional:2|this:2|and:2|equity:1|introductory:1|.head:1|pouch:1|ful:1 || L24:4086 ,:3|marvel:1|play:1|metrics:1|framework:1|time:1|BI:1|images:1
- Model counts: L16:1349 gemini-3.5-flash:7|human:1|gpt-5.5:1 || L19:536 qwen3.6-35b:15|glm-5.2:11|gpt-5.5:10|human:9|gemini-3.5-flash:5 || L20:748 qwen3.6-35b:11|gpt-5.5:9|human:8|gemini-3.5-flash:5|glm-5.2:5 || L24:4086 qwen3.6-35b:26|gemini-3.5-flash:5|glm-5.2:5|human:2

## 18. `L16:2672 -> L19:1325 -> L20:2290 -> L24:2940`

- Lexical score: `4.045`
- Edge geomean: `2.766`; edge scores `0.576|15.454|2.376`; shared prompts `3|22|5`
- Word/punctuation: min word `1.000`, max punctuation `0.000`, max period `0.000`
- Source balance: mean AI `0.718`, min AI `0.438`, mean human `0.282`
- Joined/missing docs: min joined `32`, max missing `18`
- Node ranks: `118|34|86|4`
- Peak tokens: L16:2672 C:24|c:5|Ca:1|CS:1|Cit:1 || L19:1325 light:2|mark:2|ulation:1|fountain:1|blood:1|heels:1|day:1|full:1 || L20:2290 place:2|ulation:1|blur:1|flesh:1|replaced:1|teachers:1|otion:1|drying:1 || L24:2940 the:13|The:9|a:7|A:3|my:1|its:1
- Model counts: L16:2672 human:18|gemini-3.5-flash:7|gpt-5.5:7 || L19:1325 glm-5.2:14|gpt-5.5:12|gemini-3.5-flash:9|qwen3.6-35b:9|human:6 || L20:2290 glm-5.2:11|gemini-3.5-flash:6|qwen3.6-35b:6|human:5|gpt-5.5:5 || L24:2940 human:10|glm-5.2:9|gemini-3.5-flash:8|qwen3.6-35b:5|gpt-5.5:2

## 19. `L16:1975 -> L19:33 -> L20:2290 -> L24:2940`

- Lexical score: `4.043`
- Edge geomean: `2.315`; edge scores `1.526|3.423|2.376`; shared prompts `6|9|5`
- Word/punctuation: min word `0.938`, max punctuation `0.062`, max period `0.000`
- Source balance: mean AI `0.728`, min AI `0.438`, mean human `0.272`
- Joined/missing docs: min joined `16`, max missing `34`
- Node ranks: `61|36|86|4`
- Peak tokens: L16:1975 did:1|me:1|shoes:1|mark:1|opes:1|tricks:1|ener:1|reach:1 || L19:33 and:22|are:3|is:2|so:2|to:2|also:1|cannot:1|lamp:1 || L20:2290 place:2|ulation:1|blur:1|flesh:1|replaced:1|teachers:1|otion:1|drying:1 || L24:2940 the:13|The:9|a:7|A:3|my:1|its:1
- Model counts: L16:1975 human:9|gpt-5.5:6|gemini-3.5-flash:1 || L19:33 glm-5.2:14|gemini-3.5-flash:13|qwen3.6-35b:11|gpt-5.5:8|human:4 || L20:2290 glm-5.2:11|gemini-3.5-flash:6|qwen3.6-35b:6|human:5|gpt-5.5:5 || L24:2940 human:10|glm-5.2:9|gemini-3.5-flash:8|qwen3.6-35b:5|gpt-5.5:2

## 20. `L16:2515 -> L19:1437 -> L20:1382 -> L24:1664`

- Lexical score: `4.021`
- Edge geomean: `2.627`; edge scores `0.671|3.901|6.924`; shared prompts `5|8|12`
- Word/punctuation: min word `0.959`, max punctuation `0.041`, max period `0.000`
- Source balance: mean AI `0.626`, min AI `0.143`, mean human `0.374`
- Joined/missing docs: min joined `23`, max missing `27`
- Node ranks: `91|38|11|19`
- Peak tokens: L16:2515 In:26|For:2 || L19:1437 such:5|don:3|a:3|doesn:2|kind:2|do:2|by:2|hasn:1 || L20:1382 a:3|is:2|that:2|to:1|feel:1|still:1|Easy:1|can:1 || L24:1664 to:5|can:4|is:2|still:1|feel:1|methods:1|feels:1|people:1
- Model counts: L16:2515 human:24|gpt-5.5:3|gemini-3.5-flash:1 || L19:1437 human:25|gpt-5.5:19|gemini-3.5-flash:3|glm-5.2:2 || L20:1382 gpt-5.5:19|gemini-3.5-flash:1|qwen3.6-35b:1|glm-5.2:1|human:1 || L24:1664 gpt-5.5:20|human:2|qwen3.6-35b:1
