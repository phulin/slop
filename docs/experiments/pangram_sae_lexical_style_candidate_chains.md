# Pangram SAE Lexical/Style Chain Queue

This queue filters the broad cross-layer candidate graph by peak-token distributions. It is intended to find lexical/style candidates rather than period or sentence-boundary accumulators.

## 1. `L16:689 -> L19:3990 -> L20:1210 -> L24:1120`

- Lexical score: `38.150`
- Edge geomean: `11.618`; edge scores `3.309|19.289|24.564`; shared prompts `7|24|28`
- Word/punctuation: min word `0.840`, max punctuation `0.160`, max period `0.033`
- Source balance: mean AI `0.171`, min AI `0.133`, mean human `0.829`
- Joined/missing docs: min joined `18`, max missing `32`
- Node ranks: `60|4|5|14`
- Peak tokens: L16:689 such:2|of:2|the:2|part:1|However:1|want:1|variety:1|United:1 || L19:3990 and:8|,:7|to:4|It:3|these:2|They:2|the:2|I:2 || L20:1210 and:6|to:4|the:2|their:1|was:1|his:1|a:1|ing:1 || L24:1120 of:2|it:2|to:2|you:2|is:2|anxiety:1|and:1|for:1
- Model counts: L16:689 human:14|gpt-5.5:3|gemini-3.5-flash:1 || L19:3990 human:42|gpt-5.5:8 || L20:1210 human:25|gpt-5.5:5 || L24:1120 human:26|gpt-5.5:4

## 2. `L16:2196 -> L19:3990 -> L20:1210 -> L24:1120`

- Lexical score: `22.744`
- Edge geomean: `7.390`; edge scores `0.852|19.289|24.564`; shared prompts `3|24|28`
- Word/punctuation: min word `0.840`, max punctuation `0.160`, max period `0.033`
- Source balance: mean AI `0.285`, min AI `0.133`, mean human `0.715`
- Joined/missing docs: min joined `28`, max missing `22`
- Node ranks: `93|4|5|14`
- Peak tokens: L16:2196 cir:2|Ret:1|t:1|rop:1|rap:1|w:1|ina:1|C:1 || L19:3990 and:8|,:7|to:4|It:3|these:2|They:2|the:2|I:2 || L20:1210 and:6|to:4|the:2|their:1|was:1|his:1|a:1|ing:1 || L24:1120 of:2|it:2|to:2|you:2|is:2|anxiety:1|and:1|for:1
- Model counts: L16:2196 gpt-5.5:10|human:9|gemini-3.5-flash:9 || L19:3990 human:42|gpt-5.5:8 || L20:1210 human:25|gpt-5.5:5 || L24:1120 human:26|gpt-5.5:4

## 3. `L16:642 -> L19:4063 -> L20:1373 -> L24:3645`

- Lexical score: `18.974`
- Edge geomean: `13.950`; edge scores `13.922|19.579|9.960`; shared prompts `20|24|15`
- Word/punctuation: min word `0.885`, max punctuation `0.115`, max period `0.038`
- Source balance: mean AI `0.078`, min AI `0.000`, mean human `0.922`
- Joined/missing docs: min joined `26`, max missing `24`
- Node ranks: `19|33|20|76`
- Peak tokens: L16:642 Illinois:1|all:1|ugly:1|Well:1|while:1|friendly:1|skin:1|):1 || L19:4063 said:4|it:3|you:2|ugly:1|stressful:1|54:1|all:1|shouted:1 || L20:1373 said:2|like:2|read:1|ugly:1|it:1|0:1|Illinois:1|plate:1 || L24:3645 together:1|enough:1|ship:1|up:1|power:1|route:1|mistake:1|ago:1
- Model counts: L16:642 human:26 || L19:4063 human:44|glm-5.2:6 || L20:1373 human:25|glm-5.2:6 || L24:3645 human:37

## 4. `L16:3450 -> L19:3990 -> L20:1210 -> L24:1120`

- Lexical score: `14.518`
- Edge geomean: `4.534`; edge scores `0.197|19.289|24.564`; shared prompts `4|24|28`
- Word/punctuation: min word `0.840`, max punctuation `0.160`, max period `0.033`
- Source balance: mean AI `0.352`, min AI `0.133`, mean human `0.648`
- Joined/missing docs: min joined `19`, max missing `31`
- Node ranks: `18|4|5|14`
- Peak tokens: L16:3450 such:5|you:2|align:2|-:2|Additionally:2|of:1|one:1|known:1 || L19:3990 and:8|,:7|to:4|It:3|these:2|They:2|the:2|I:2 || L20:1210 and:6|to:4|the:2|their:1|was:1|his:1|a:1|ing:1 || L24:1120 of:2|it:2|to:2|you:2|is:2|anxiety:1|and:1|for:1
- Model counts: L16:3450 gemini-3.5-flash:13|gpt-5.5:5|human:1 || L19:3990 human:42|gpt-5.5:8 || L20:1210 human:25|gpt-5.5:5 || L24:1120 human:26|gpt-5.5:4

## 5. `L16:3800 -> L19:262 -> L20:1382 -> L24:1664`

- Lexical score: `13.920`
- Edge geomean: `6.475`; edge scores `4.396|8.919|6.924`; shared prompts `9|15|12`
- Word/punctuation: min word `0.889`, max punctuation `0.111`, max period `0.037`
- Source balance: mean AI `0.779`, min AI `0.444`, mean human `0.221`
- Joined/missing docs: min joined `23`, max missing `27`
- Node ranks: `92|9|11|19`
- Peak tokens: L16:3800 don:2|by:2|such:2|each:1|.:1|lot:1|However:1|kinds:1 || L19:262 a:7|to:4|are:2|might:2|can:2|is:2|don:2|that:2 || L20:1382 a:3|is:2|that:2|to:1|feel:1|still:1|Easy:1|can:1 || L24:1664 to:5|can:4|is:2|still:1|feel:1|methods:1|feels:1|people:1
- Model counts: L16:3800 human:15|gpt-5.5:11|gemini-3.5-flash:1 || L19:262 gpt-5.5:33|human:10|glm-5.2:3|gemini-3.5-flash:2|qwen3.6-35b:2 || L20:1382 gpt-5.5:19|gemini-3.5-flash:1|qwen3.6-35b:1|glm-5.2:1|human:1 || L24:1664 gpt-5.5:20|human:2|qwen3.6-35b:1

## 6. `L16:3800 -> L19:1437 -> L20:1382 -> L24:1664`

- Lexical score: `12.420`
- Edge geomean: `7.199`; edge scores `13.814|3.901|6.924`; shared prompts `24|8|12`
- Word/punctuation: min word `0.889`, max punctuation `0.111`, max period `0.037`
- Source balance: mean AI `0.701`, min AI `0.444`, mean human `0.299`
- Joined/missing docs: min joined `23`, max missing `27`
- Node ranks: `92|38|11|19`
- Peak tokens: L16:3800 don:2|by:2|such:2|each:1|.:1|lot:1|However:1|kinds:1 || L19:1437 such:5|don:3|a:3|doesn:2|kind:2|do:2|by:2|hasn:1 || L20:1382 a:3|is:2|that:2|to:1|feel:1|still:1|Easy:1|can:1 || L24:1664 to:5|can:4|is:2|still:1|feel:1|methods:1|feels:1|people:1
- Model counts: L16:3800 human:15|gpt-5.5:11|gemini-3.5-flash:1 || L19:1437 human:25|gpt-5.5:19|gemini-3.5-flash:3|glm-5.2:2 || L20:1382 gpt-5.5:19|gemini-3.5-flash:1|qwen3.6-35b:1|glm-5.2:1|human:1 || L24:1664 gpt-5.5:20|human:2|qwen3.6-35b:1

## 7. `L16:689 -> L19:3990 -> L20:1210 -> L24:2776`

- Lexical score: `10.189`
- Edge geomean: `4.213`; edge scores `3.309|19.289|1.171`; shared prompts `7|24|3`
- Word/punctuation: min word `0.840`, max punctuation `0.160`, max period `0.033`
- Source balance: mean AI `0.365`, min AI `0.160`, mean human `0.635`
- Joined/missing docs: min joined `18`, max missing `32`
- Node ranks: `60|4|5|95`
- Peak tokens: L16:689 such:2|of:2|the:2|part:1|However:1|want:1|variety:1|United:1 || L19:3990 and:8|,:7|to:4|It:3|these:2|They:2|the:2|I:2 || L20:1210 and:6|to:4|the:2|their:1|was:1|his:1|a:1|ing:1 || L24:2776 with:5|requires:5|a:5|to:3|is:2|require:2|involves:1|and:1
- Model counts: L16:689 human:14|gpt-5.5:3|gemini-3.5-flash:1 || L19:3990 human:42|gpt-5.5:8 || L20:1210 human:25|gpt-5.5:5 || L24:2776 qwen3.6-35b:15|gpt-5.5:8|glm-5.2:4|gemini-3.5-flash:4|human:3

## 8. `L16:2989 -> L19:2834 -> L20:1373 -> L24:3645`

- Lexical score: `8.655`
- Edge geomean: `6.832`; edge scores `11.982|2.672|9.960`; shared prompts `17|5|15`
- Word/punctuation: min word `0.970`, max punctuation `0.030`, max period `0.030`
- Source balance: mean AI `0.048`, min AI `0.000`, mean human `0.952`
- Joined/missing docs: min joined `31`, max missing `19`
- Node ranks: `40|45|20|76`
- Peak tokens: L16:2989 in:3|form:1|beneficial:1|effect:1|himself:1|s:1|follow:1|benefit:1 || L19:2834 is:6|to:5|a:3|will:3|would:2|the:2|you:2|that:2 || L20:1373 said:2|like:2|read:1|ugly:1|it:1|0:1|Illinois:1|plate:1 || L24:3645 together:1|enough:1|ship:1|up:1|power:1|route:1|mistake:1|ago:1
- Model counts: L16:2989 human:33 || L19:2834 human:50 || L20:1373 human:25|glm-5.2:6 || L24:3645 human:37

## 9. `L16:689 -> L19:3990 -> L20:1210 -> L24:3469`

- Lexical score: `8.528`
- Edge geomean: `3.515`; edge scores `3.309|19.289|0.680`; shared prompts `7|24|4`
- Word/punctuation: min word `0.840`, max punctuation `0.160`, max period `0.043`
- Source balance: mean AI `0.344`, min AI `0.160`, mean human `0.656`
- Joined/missing docs: min joined `18`, max missing `32`
- Node ranks: `60|4|5|47`
- Peak tokens: L16:689 such:2|of:2|the:2|part:1|However:1|want:1|variety:1|United:1 || L19:3990 and:8|,:7|to:4|It:3|these:2|They:2|the:2|I:2 || L20:1210 and:6|to:4|the:2|their:1|was:1|his:1|a:1|ing:1 || L24:3469 such:7|However:4|Additionally:2|Furthermore:2|you:2|United:1|.:1|isn:1
- Model counts: L16:689 human:14|gpt-5.5:3|gemini-3.5-flash:1 || L19:3990 human:42|gpt-5.5:8 || L20:1210 human:25|gpt-5.5:5 || L24:3469 gpt-5.5:8|qwen3.6-35b:7|glm-5.2:4|human:4

## 10. `L16:1975 -> L19:1146 -> L20:3702 -> L24:2940`

- Lexical score: `8.352`
- Edge geomean: `3.157`; edge scores `0.824|8.042|4.747`; shared prompts `3|14|9`
- Word/punctuation: min word `0.840`, max punctuation `0.160`, max period `0.120`
- Source balance: mean AI `0.514`, min AI `0.340`, mean human `0.486`
- Joined/missing docs: min joined `16`, max missing `34`
- Node ranks: `61|7|3|4`
- Peak tokens: L16:1975 did:1|me:1|shoes:1|mark:1|opes:1|tricks:1|ener:1|reach:1 || L19:1146 The:14|the:9|.:6|and:4|a:4|with:2|A:2|They:2 || L20:3702 The:19|the:7|A:1|a:1 || L24:2940 the:13|The:9|a:7|A:3|my:1|its:1
- Model counts: L16:1975 human:9|gpt-5.5:6|gemini-3.5-flash:1 || L19:1146 human:33|glm-5.2:6|qwen3.6-35b:5|gemini-3.5-flash:4|gpt-5.5:2 || L20:3702 human:12|glm-5.2:6|gemini-3.5-flash:5|qwen3.6-35b:3|gpt-5.5:2 || L24:2940 human:10|glm-5.2:9|gemini-3.5-flash:8|qwen3.6-35b:5|gpt-5.5:2

## 11. `L16:1975 -> L19:1325 -> L20:2290 -> L24:2940`

- Lexical score: `7.822`
- Edge geomean: `4.456`; edge scores `2.410|15.454|2.376`; shared prompts `9|22|5`
- Word/punctuation: min word `0.938`, max punctuation `0.062`, max period `0.000`
- Source balance: mean AI `0.718`, min AI `0.438`, mean human `0.282`
- Joined/missing docs: min joined `16`, max missing `34`
- Node ranks: `61|34|86|4`
- Peak tokens: L16:1975 did:1|me:1|shoes:1|mark:1|opes:1|tricks:1|ener:1|reach:1 || L19:1325 light:2|mark:2|ulation:1|fountain:1|blood:1|heels:1|day:1|full:1 || L20:2290 place:2|ulation:1|blur:1|flesh:1|replaced:1|teachers:1|otion:1|drying:1 || L24:2940 the:13|The:9|a:7|A:3|my:1|its:1
- Model counts: L16:1975 human:9|gpt-5.5:6|gemini-3.5-flash:1 || L19:1325 glm-5.2:14|gpt-5.5:12|gemini-3.5-flash:9|qwen3.6-35b:9|human:6 || L20:2290 glm-5.2:11|gemini-3.5-flash:6|qwen3.6-35b:6|human:5|gpt-5.5:5 || L24:2940 human:10|glm-5.2:9|gemini-3.5-flash:8|qwen3.6-35b:5|gpt-5.5:2

## 12. `L16:3450 -> L19:3450 -> L20:1210 -> L24:1120`

- Lexical score: `7.813`
- Edge geomean: `3.155`; edge scores `1.601|0.799|24.564`; shared prompts `6|4|28`
- Word/punctuation: min word `0.895`, max punctuation `0.105`, max period `0.042`
- Source balance: mean AI `0.531`, min AI `0.133`, mean human `0.469`
- Joined/missing docs: min joined `19`, max missing `31`
- Node ranks: `18|119|5|14`
- Peak tokens: L16:3450 such:5|you:2|align:2|-:2|Additionally:2|of:1|one:1|known:1 || L19:3450 However:20|such:7|Furthermore:3|.:2|you:2|Additionally:1|type:1|however:1 || L20:1210 and:6|to:4|the:2|their:1|was:1|his:1|a:1|ing:1 || L24:1120 of:2|it:2|to:2|you:2|is:2|anxiety:1|and:1|for:1
- Model counts: L16:3450 gemini-3.5-flash:13|gpt-5.5:5|human:1 || L19:3450 qwen3.6-35b:22|glm-5.2:9|human:6|gpt-5.5:6|gemini-3.5-flash:5 || L20:1210 human:25|gpt-5.5:5 || L24:1120 human:26|gpt-5.5:4

## 13. `L16:1975 -> L19:1325 -> L20:3702 -> L24:2940`

- Lexical score: `7.343`
- Edge geomean: `3.056`; edge scores `2.410|2.496|4.747`; shared prompts `9|5|9`
- Word/punctuation: min word `0.938`, max punctuation `0.062`, max period `0.000`
- Source balance: mean AI `0.649`, min AI `0.438`, mean human `0.351`
- Joined/missing docs: min joined `16`, max missing `34`
- Node ranks: `61|34|3|4`
- Peak tokens: L16:1975 did:1|me:1|shoes:1|mark:1|opes:1|tricks:1|ener:1|reach:1 || L19:1325 light:2|mark:2|ulation:1|fountain:1|blood:1|heels:1|day:1|full:1 || L20:3702 The:19|the:7|A:1|a:1 || L24:2940 the:13|The:9|a:7|A:3|my:1|its:1
- Model counts: L16:1975 human:9|gpt-5.5:6|gemini-3.5-flash:1 || L19:1325 glm-5.2:14|gpt-5.5:12|gemini-3.5-flash:9|qwen3.6-35b:9|human:6 || L20:3702 human:12|glm-5.2:6|gemini-3.5-flash:5|qwen3.6-35b:3|gpt-5.5:2 || L24:2940 human:10|glm-5.2:9|gemini-3.5-flash:8|qwen3.6-35b:5|gpt-5.5:2

## 14. `L16:1975 -> L19:1325 -> L20:2290 -> L24:457`

- Lexical score: `7.189`
- Edge geomean: `5.998`; edge scores `2.410|15.454|5.794`; shared prompts `9|22|12`
- Word/punctuation: min word `0.938`, max punctuation `0.062`, max period `0.000`
- Source balance: mean AI `0.791`, min AI `0.438`, mean human `0.209`
- Joined/missing docs: min joined `16`, max missing `34`
- Node ranks: `61|34|86|107`
- Peak tokens: L16:1975 did:1|me:1|shoes:1|mark:1|opes:1|tricks:1|ener:1|reach:1 || L19:1325 light:2|mark:2|ulation:1|fountain:1|blood:1|heels:1|day:1|full:1 || L20:2290 place:2|ulation:1|blur:1|flesh:1|replaced:1|teachers:1|otion:1|drying:1 || L24:457 bright:3|grace:3|tall:2|frame:2|high:2|flight:1|wide:1|way:1
- Model counts: L16:1975 human:9|gpt-5.5:6|gemini-3.5-flash:1 || L19:1325 glm-5.2:14|gpt-5.5:12|gemini-3.5-flash:9|qwen3.6-35b:9|human:6 || L20:2290 glm-5.2:11|gemini-3.5-flash:6|qwen3.6-35b:6|human:5|gpt-5.5:5 || L24:457 qwen3.6-35b:10|gemini-3.5-flash:10|glm-5.2:7|gpt-5.5:3

## 15. `L16:3085 -> L19:420 -> L20:1210 -> L24:1120`

- Lexical score: `6.752`
- Edge geomean: `2.687`; edge scores `0.609|1.297|24.564`; shared prompts `3|3|28`
- Word/punctuation: min word `0.929`, max punctuation `0.071`, max period `0.033`
- Source balance: mean AI `0.506`, min AI `0.133`, mean human `0.494`
- Joined/missing docs: min joined `28`, max missing `22`
- Node ranks: `56|88|5|14`
- Peak tokens: L16:3085 is:5|,:2|remains:2|saved:1|Cafe:1|also:1|into:1|by:1 || L19:420 a:14|poignant:3|case:2|serves:2|serve:2|it:2|,:2|be:2 || L20:1210 and:6|to:4|the:2|their:1|was:1|his:1|a:1|ing:1 || L24:1120 of:2|it:2|to:2|you:2|is:2|anxiety:1|and:1|for:1
- Model counts: L16:3085 gpt-5.5:26|gemini-3.5-flash:1|human:1 || L19:420 qwen3.6-35b:17|human:12|glm-5.2:9|gpt-5.5:7|gemini-3.5-flash:5 || L20:1210 human:25|gpt-5.5:5 || L24:1120 human:26|gpt-5.5:4

## 16. `L16:3260 -> L19:1459 -> L20:2956 -> L24:1599`

- Lexical score: `6.552`
- Edge geomean: `6.462`; edge scores `3.431|7.540|10.431`; shared prompts `12|17|20`
- Word/punctuation: min word `0.905`, max punctuation `0.095`, max period `0.095`
- Source balance: mean AI `0.166`, min AI `0.048`, mean human `0.834`
- Joined/missing docs: min joined `21`, max missing `29`
- Node ranks: `80|39|87|100`
- Peak tokens: L16:3260 .:2|never:1|of:1|exactly:1|describe:1|now:1|intensely:1|compromise:1 || L19:1459 very:2|of:2|it:2|to:2|s:2|never:1|function:1|market:1 || L20:2956 a:2|never:1|of:1|he:1|ultimate:1|it:1|well:1|not:1 || L24:1599 it:2|was:1|are:1|will:1|who:1|up:1|to:1|great:1
- Model counts: L16:3260 human:20|gemini-3.5-flash:1 || L19:1459 human:41|glm-5.2:4|gemini-3.5-flash:3|qwen3.6-35b:2 || L20:2956 human:17|glm-5.2:3|qwen3.6-35b:2|gpt-5.5:1|gemini-3.5-flash:1 || L24:1599 human:18|glm-5.2:1|qwen3.6-35b:1|gpt-5.5:1

## 17. `L16:3085 -> L19:3450 -> L20:1210 -> L24:1120`

- Lexical score: `6.415`
- Edge geomean: `2.621`; edge scores `0.918|0.799|24.564`; shared prompts `3|4|28`
- Word/punctuation: min word `0.929`, max punctuation `0.071`, max period `0.042`
- Source balance: mean AI `0.535`, min AI `0.133`, mean human `0.465`
- Joined/missing docs: min joined `28`, max missing `22`
- Node ranks: `56|119|5|14`
- Peak tokens: L16:3085 is:5|,:2|remains:2|saved:1|Cafe:1|also:1|into:1|by:1 || L19:3450 However:20|such:7|Furthermore:3|.:2|you:2|Additionally:1|type:1|however:1 || L20:1210 and:6|to:4|the:2|their:1|was:1|his:1|a:1|ing:1 || L24:1120 of:2|it:2|to:2|you:2|is:2|anxiety:1|and:1|for:1
- Model counts: L16:3085 gpt-5.5:26|gemini-3.5-flash:1|human:1 || L19:3450 qwen3.6-35b:22|glm-5.2:9|human:6|gpt-5.5:6|gemini-3.5-flash:5 || L20:1210 human:25|gpt-5.5:5 || L24:1120 human:26|gpt-5.5:4

## 18. `L16:1975 -> L19:1325 -> L20:2290 -> L24:43`

- Lexical score: `6.402`
- Edge geomean: `4.937`; edge scores `2.410|15.454|3.231`; shared prompts `9|22|6`
- Word/punctuation: min word `0.938`, max punctuation `0.062`, max period `0.000`
- Source balance: mean AI `0.708`, min AI `0.438`, mean human `0.292`
- Joined/missing docs: min joined `16`, max missing `34`
- Node ranks: `61|34|86|18`
- Peak tokens: L16:1975 did:1|me:1|shoes:1|mark:1|opes:1|tricks:1|ener:1|reach:1 || L19:1325 light:2|mark:2|ulation:1|fountain:1|blood:1|heels:1|day:1|full:1 || L20:2290 place:2|ulation:1|blur:1|flesh:1|replaced:1|teachers:1|otion:1|drying:1 || L24:43 way:3|be:1|breath:1|pick:1|stay:1|rust:1|yesterday:1|course:1
- Model counts: L16:1975 human:9|gpt-5.5:6|gemini-3.5-flash:1 || L19:1325 glm-5.2:14|gpt-5.5:12|gemini-3.5-flash:9|qwen3.6-35b:9|human:6 || L20:2290 glm-5.2:11|gemini-3.5-flash:6|qwen3.6-35b:6|human:5|gpt-5.5:5 || L24:43 gpt-5.5:10|human:10|gemini-3.5-flash:5|glm-5.2:3|qwen3.6-35b:2

## 19. `L16:689 -> L19:3450 -> L20:1210 -> L24:1120`

- Lexical score: `6.334`
- Edge geomean: `2.476`; edge scores `0.774|0.799|24.564`; shared prompts `3|4|28`
- Word/punctuation: min word `0.933`, max punctuation `0.067`, max period `0.042`
- Source balance: mean AI `0.349`, min AI `0.133`, mean human `0.651`
- Joined/missing docs: min joined `18`, max missing `32`
- Node ranks: `60|119|5|14`
- Peak tokens: L16:689 such:2|of:2|the:2|part:1|However:1|want:1|variety:1|United:1 || L19:3450 However:20|such:7|Furthermore:3|.:2|you:2|Additionally:1|type:1|however:1 || L20:1210 and:6|to:4|the:2|their:1|was:1|his:1|a:1|ing:1 || L24:1120 of:2|it:2|to:2|you:2|is:2|anxiety:1|and:1|for:1
- Model counts: L16:689 human:14|gpt-5.5:3|gemini-3.5-flash:1 || L19:3450 qwen3.6-35b:22|glm-5.2:9|human:6|gpt-5.5:6|gemini-3.5-flash:5 || L20:1210 human:25|gpt-5.5:5 || L24:1120 human:26|gpt-5.5:4

## 20. `L16:689 -> L19:3990 -> L20:1210 -> L24:3009`

- Lexical score: `6.217`
- Edge geomean: `2.356`; edge scores `3.309|19.289|0.205`; shared prompts `7|24|3`
- Word/punctuation: min word `0.840`, max punctuation `0.160`, max period `0.040`
- Source balance: mean AI `0.247`, min AI `0.160`, mean human `0.753`
- Joined/missing docs: min joined `18`, max missing `32`
- Node ranks: `60|4|5|30`
- Peak tokens: L16:689 such:2|of:2|the:2|part:1|However:1|want:1|variety:1|United:1 || L19:3990 and:8|,:7|to:4|It:3|these:2|They:2|the:2|I:2 || L20:1210 and:6|to:4|the:2|their:1|was:1|his:1|a:1|ing:1 || L24:3009 ,:2|engineers:1|result:1|ing:1|side:1|guaranteed:1|matters:1|sensation:1
- Model counts: L16:689 human:14|gpt-5.5:3|gemini-3.5-flash:1 || L19:3990 human:42|gpt-5.5:8 || L20:1210 human:25|gpt-5.5:5 || L24:3009 human:14|glm-5.2:9|qwen3.6-35b:1|gpt-5.5:1
