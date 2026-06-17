#!/usr/bin/env node
import winkPosTagger from "wink-pos-tagger";

const MAXG = 160;
const PRON = "(?:it|they|this|that)";
const BE = "(?:is|are|was|were)";
const BE_NEG =
  "(?:is\\s+not|are\\s+not|was\\s+not|were\\s+not|isn't|aren't|wasn't|weren't|ain't)";

const STAGE1_REGEXES = {
  RE_NOT_BUT: new RegExp(
    `\\b(?:(?:${BE_NEG})|not(?!\\s+(?:that|only)\\b))\\s+` +
      `(?:(?!\\bbut\\b|[.?!]).)` +
      `{1,100}?` +
      `[,;:]\\s*but\\s+` +
      `(?!when\\b|while\\b|which\\b|who\\b|whom\\b|whose\\b|where\\b|if\\b|that\\b|as\\b|because\\b|although\\b|though\\b|till\\b|until\\b|unless\\b|` +
      `here\\b|there\\b|then\\b|my\\b|we\\b|I\\b|you\\b|it\\s+seems\\b|it\\s+appears\\b|it\\s+felt\\b|it\\s+looks?\\b|anything\\b)`,
    "gi",
  ),
  RE_NOT_DASH: new RegExp(
    `\\b(?:\\w+n't|not)\\s+(?:just|only|merely)?\\s+` +
      `(?:(?![.?!]).){1,${MAXG}}?` +
      `(?:-|\\s-\\s|[\\u2014\\u2013])\\s*` +
      `${PRON}\\s+(?:(?:'re|are|'s|is|were|was)\\b|(?!'re|are|'s|is|were|was)[*_~]*[a-z]\\w*)`,
    "gi",
  ),
  RE_PRON_BE_NOT_SEP_BE: new RegExp(
    `(?:(?<=^)|(?<=[.?!]\\s))\\s*[""']?` +
      `(?:(?:${PRON}\\s+${BE}\\s+not)|(?:${PRON}\\s+${BE}n't)|(?:it's|they're|that's)\\s+not)\\b` +
      `[^.?!]{0,${MAXG}}[.;:?!]\\s*[""']?` +
      `${PRON}\\s+(?:${BE}|(?:'s|'re))\\b(?!\\s+not\\b)`,
    "gi",
  ),
  RE_NP_BE_NOT_SEP_THEY_BE: new RegExp(
    `(?:(?<=^)|(?<=[.?!]\\s))\\s*` +
      `(?![^.?!]{0,80}\\b(?:knew|know|thought|think|said|says|told|heard|learned)\\b[^.?!]{0,40}?\\bthat\\b)` +
      `(?!\\s*not\\s+without\\b)` +
      `(?![^.?!]{0,50}\\bnot\\s+put\\b)` +
      `[^.?!]{0,${MAXG}}?\\b(?:${BE_NEG})\\b[^.?!]{0,${MAXG}}[.;:?!]\\s*` +
      `[""']?${PRON}\\b(?:'re|\\s+(?:are|were|is|was))\\b(?!\\s+not\\b)`,
    "gi",
  ),
  RE_NO_LONGER: new RegExp(
    `(?:(?<=^)|(?<=[.?!]\\s))\\s*[^.?!]{0,${MAXG}}\\bno\\s+longer\\b[^.;:?!]{0,${MAXG}}` +
      `[.;:?!]\\s*(?:it|they|this|that)\\s+(?:is|are|was|were)\\b(?!\\s+not\\b)`,
    "gi",
  ),
  RE_NOT_JUST_SEP: new RegExp(
    `(?:(?<=^)|(?<=[.?!]\\s))\\s*[""']?` +
      `${PRON}\\b(?:'s|'re|\\s+(?:is|are|was|were))?\\s+not\\s+just\\b[^.?!]{0,${MAXG}}[.?!]\\s*[""']?` +
      `${PRON}\\b(?:'s|'re|\\s+(?:is|are|was|were))\\b(?!\\s+not\\b)`,
    "gi",
  ),
  RE_NOT_PERIOD_SAMEVERB: new RegExp(
    `(?:(?<=^)|(?<=[.?!]\\s))[^.?!]*?\\b(?:do|does|did)n't\\b\\s+` +
      `(?:(?:\\w+\\s+){0,2})([a-z]{3,})\\b[^.?!]*[.?!]\\s*` +
      `${PRON}\\s+\\1(?:ed|es|s|ing)?\\b`,
    "gi",
  ),
  RE_SIMPLE_BE_NOT_IT_BE: new RegExp(
    `(?:(?<=^)|(?<=[.?!]\\s))\\s*[""']?` +
      `(?!he\\b|she\\b|i\\b|you\\b|we\\b)` +
      `(?![^.?!]{0,80}\\b(?:knew|know|thought|think|said|says|told|heard|learned)\\b[^.?!]{0,40}?\\bthat\\b)` +
      `[^.?!]{0,${MAXG}}?\\b${BE_NEG}\\b[^.?!]{0,${MAXG}}[.;:?!]\\s*` +
      `[""']?it(?:'s|\\s+(?:is|are|was|were))\\b`,
    "gi",
  ),
  RE_EMBEDDED_NOT_JUST_SEP: new RegExp(
    `(?:(?<=^)|(?<=[.?!]\\s))` +
      `[^.?!]{0,80}?\\b(?:(?:it|they)\\s+(?:is|are)|(?:it's|they're))\\s+not\\s+just\\b` +
      `[^.?!]{0,${MAXG}}[.?!]\\s*` +
      `(?:(?:it|they)\\s+(?:is|are)|(?:it's|they're))\\b`,
    "gi",
  ),
  RE_DIALOGUE_NOT_JUST: new RegExp(
    `[""']?${PRON}(?:'re|'s|\\s+(?:are|is|was|were))\\s+not\\s+just\\b[^""']{0,${MAXG}}[""']?\\s*` +
      `(?:[^.?!]{0,80}\\b(?:said|asked|whispered|muttered|replied|added|shouted|cried)\\b[^.?!]{0,80}[.?!]\\s*)?` +
      `[""']?${PRON}(?:'re|'s|\\s+(?:are|is|was|were))\\s+[*_~]?[a-z]\\w*`,
    "gi",
  ),
};

const STAGE2_REGEXES = {
  POS_DOESNT_VERB:
    /["'"]\s*(?:[Tt]he\s+\w+|[Ii]t|[Tt]hey|[Yy]ou)\s+doesn't\s+VERB[^.!?]*?[.!?]\s*(?:it|they|you|that)\s+[*_~]?(?:VERB|whispers?|reminds?|signals?|tests?|speaks?)/gi,
  POS_DONT_JUST_VERB:
    /["'"]\s*(?:[Tt]hey|[Yy]ou|[Ii]t)\s+don't\s+just\s+VERB[^.!?]*?[—-]\s*they\s+[*_~]?VERB/gi,
  POS_GERUND_FRAGMENT: /["'"]\s*Not\s+just\s+VERB[.!?]\s+[*_~]?VERB[.!?]/g,
  POS_NOT_ADJ:
    /\bnot\s+(random|passive|simple|normal)[.!?;]\s+(?:[Tt]hey|[Ii]t)\s+(?:were|was|are|is|'re|'s)\s+[*_~]?(intentional|active|complex|different|\w{8,})/gi,
  POS_DASH_VERB:
    /\b(?:wasn't|weren't|isn't|aren't)\s+just\s+(?:VERB|a\s+\w+)[^-]{0,30}?-\s*(?:it|they)\s+(?:was|were|is|are|'s|'re)\s+[*_~]?(?:VERB|a\s+[*_~]?\w+)/gi,
  POS_NOT_JUST_VERB_PAST:
    /\b(?:was|were)\s+not\s+just\s+(?:VERB|a\s+\w+)[.!?]\s+(?:[Ii]t|[Tt]hey)\s+(?:was|were)\s+[*_~]?(?:VERB|a\s+[*_~]?\w+)/gi,
  POS_COLON_VERB:
    /:\s+(?:the\s+\w+|it|they)\s+(?:was|were)\s+not\s+just\s+VERB[.!?]\s+(?:[Ii]t|[Tt]hey)\s+(?:was|were)\s+[*_~]?VERB/gi,
  POS_ISNT_JUST_VERB:
    /["'"]\s*(?:[^"'"]{0,100}?\b)?(?:The\s+\w+|It|They|You)\s+(?:isn't|aren't|wasn't|weren't)\s+just\s+VERB[^"'".!?]{0,40}?[—-]\s*(?:it's|they're)\s+[*_~]?VERB/gi,
  POS_QUOTE_MULTI_VERB:
    /["'"]\s*[^"'"]{0,150}?\b(?:not\s+just|isn't|aren't)\s+(?:VERB|a\s+\w+)[^"'".!?]{0,60}?[.!?]\s+(?:[^"'"]{0,40}?\b)?(?:It's|They're|You're|That's)\s+[*_~]?(?:VERB|a\s+[*_~]?\w+)/gi,
  POS_ELLIPSIS_VERB:
    /["'"]\s*[^"'"]{0,100}?\b(?:not\s+just|isn't)\s+VERB[^"'"]{0,30}?[.…]\s*[.…]\s*(?:they're|it's|you're)\s+[*_~]?VERB/gi,
  POS_NOT_NOUN:
    /["'"]\s*(?:That's|It's)\s+not\s+(?:a\s+)?(sign|message|warning|pattern|test|phenomenon|one\s+\w+)[.!?]\s+(?:That's|It's)\s+(?:a\s+|\*?all\s+)?[*_~]?(warning|question|language|symbol|test|presence|story|challenge|\w+)/gi,
  POS_DOESNT_VERB_EMPHASIS:
    /["'"]\s*(?:The\s+\w+|It|They)\s+doesn't\s+(?:VERB|react|warn|speak)[.!?]\s+It\s+\*(?:VERB|whispers?|reminds?|signals?)/gi,
  POS_DASH_VERB_BROAD:
    /\b(?:wasn't|weren't|isn't|aren't|don't|doesn't)\s+just\s+(?:VERB|(?:the|a)\s+\w+)[^-]{0,40}?-\s*(?:it|they)\s+(?:was|were|is|are|'s|'re)?\s*[*_~]?(?:VERB|(?:the|a)\s+[*_~]?\w+)/gi,
  POS_ELLIPSIS_BROAD:
    /["'"]\s*(?:[^"'"]{0,100}?\b)?(?:They're|You're|This)\s+(?:not\s+just|isn't)\s+(?:VERB|a\s+\w+)[^"'"]{0,40}?[.…]\s*[.…]\s*(?:they're|it's|you're|this)\s+(?:VERB|a\s+\w+)/gi,
  POS_NOT_BECAUSE: /\bit's\s+not\s+because\s+[^.!?]{5,60}?[.!?]\s+(?:It's|That's)\s+because\s+[^.!?]{5,60}/gi,
  POS_GERUND_BROAD: /["'"]\s*Not\s+just\s+VERB[.!?]\s+\*VERB[.!?]?/g,
  POS_QUOTE_VERBING:
    /["'"]\s*(?:You're|They're|It's)\s+not\s+(?:just\s+)?VERB[^"'".!?]{0,30}?[.,]\s+[^"'"]{0,50}?(?:You're|They're|It's)\s+(?:VERB|waiting)/gi,
  POS_DOESNT_LITERAL:
    /["'"]\s*(?:The\s+\w+|It|They)\s+doesn't\s+(?:VERB|react|warn|speak|listen)\s*[.!?]\s+It\s+\*\w+\*/gi,
  POS_DASH_NOUN_SWAP:
    /\b(?:was|were|is|are)\s+not\s+just\s+a\s+\w+[^-]{0,10}?-\s*(?:it|they)\s+(?:was|were|is|are)\s+(?:a\s+)?\*\w+\*/gi,
  POS_ISNT_DASH_EMPHASIS:
    /["'"]\s*(?:The\s+\w+|It|They)\s+(?:isn't|aren't|wasn't|weren't)\s+just\s+(?:VERB|a\s+\w+)[^-]{0,40}?-\s*(?:it's|they're)\s+\*\w+\*/gi,
  POS_THATS_NOT_NOUN:
    /["'"]\s*That's\s+not\s+(?:a\s+)?(?:sign|message|pattern|phenomenon|test|one\s+\w+|\w+)[.!?]\s+(?:That's|It's)\s+(?:a\s+)?\*\w+\*/gi,
  POS_GERUND_EMPHASIS: /["'"]\s*Not\s+just\s+(?:VERB|reacting|dying|\w+ing)[.!?]\s+\*[A-Z]\w+\*/g,
  POS_QUOTE_ATTRIBUTION_VERB:
    /["'"]\s*(?:The\s+\w+|They)\s+(?:are|were|'re)\s+not\s+just\s+VERB,"\s+[^"'"]{0,30}?\.\s+"They're\s+\*?VERB/gi,
  POS_ISNT_NOUN: /["'"]\s*(?:This|That|It)\s+isn't\s+just\s+a\s+\w+[.!?]\s+It's\s+(?:a\s+)?\*\w+\*/gi,
  POS_ITS_NOT_JUST: /["'"]\s*It's\s+not\s+just\s+(?:one\s+)?(\w+)[.!?]\s+It's\s+\*(?:all|every|each|\w+)\*/gi,
  POS_DASH_GERUND_OBJ:
    /["'"]\s*(?:They're|You're|It's)\s+not\s+just\s+(?:VERB|emitting|dying|\w+ing)\s+(?:a|an|the)\s+\w+[^-]{0,10}?-\s*(?:they're|you're|it's)\s+\*\w+\*/gi,
  POS_ELLIPSIS_DIALOGUE:
    /["'"]\s*(?:They're|You're|It's)\s+not\s+just\s+VERB,"\s+[^"'"]{5,40}?\.\s+"(?:They're|You're|It's)[…\s]+(?:VERB|\w+ing)/gi,
  POS_SEMI_NOUN: /\b(?:were|was|are|is)\s+not\s+just\s+(?:folklore|\w+);\s+(?:they|it)\s+(?:were|was|are|is)\s+a\s+\w+/gi,
  POS_ISNT_ADJ_NOUN:
    /["'"]\s*(?:[^"'"]{0,30}?\b)?(?:this|that|it)\s+isn't\s+just\s+a\s+(?:natural\s+)?\w+[.!?]\s+It's\s+(?:a\s+)?\*\w+\*/gi,
  POS_DIALOGUE_ATTR:
    /["'"]\s*(?:You're|They're|It's|The\s+\w+)\s+(?:(?:are|is|'re|'s)\s+)?not\s+just\s+(?:VERB(?:\s+\w+)?|a\s+\w+),"\s+[^"'"]{3,50}?\.\s+"(?:You're|They're|It's)\s+(?:a\s+)?\*\w+\*/gi,
  POS_TO_VERB_ISNT:
    /["'"]\s*To\s+VERB\s+(?:that\s+)?[^"'"]{5,50}?isn't\s+just\s+a\s+\w+[.!?]\s+It's\s+(?:a\s+)?\*\w+\*/gi,
  POS_I_AM_NOT_SEMI: /\bI\s+am\s+not\s+VERB[^;]{5,80}?;\s*it\s+is\b/gi,
  POS_NOT_ANYMORE_ITS: /\bIt's\s+not\s+[A-Z]\w+\s+anymore[.!?]\s+It's\s+[A-Z]\w+/g,
  POS_AINT_SIMPLE: /\b(?:That|This)\s+ain't\s+[^.!?]{3,40}?[.!?]\s+(?:They|It)\s+\w+/gi,
  LEMMA_SAME_VERB:
    /\b(REACT|SPEAK|LISTEN|LEARN|SIGNAL|WARN|DIE|LIVE|TEST|TEACH|AMPLIFY|INTERPRET|TRANSLATE|DECODE|EMIT)\b[^.!?]{5,80}?[.!?;—-]\s*[^.!?]{0,40}?\b\1\b/gi,
};

const VERB_TAGS = new Set(["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]);
const tagger = winkPosTagger();

function normalizeText(text) {
  return text
    .replace(/\u201c/g, '"')
    .replace(/\u201d/g, '"')
    .replace(/\u2018/g, "'")
    .replace(/\u2019/g, "'")
    .replace(/\u2014/g, "-")
    .replace(/\u2013/g, "-");
}

function sentenceSpans(text) {
  const spans = [];
  const sentSplit = /[^.!?]*[.!?]/gs;
  let lastEnd = 0;
  let match;
  while ((match = sentSplit.exec(text)) !== null) {
    spans.push([match.index, match.index + match[0].length]);
    lastEnd = match.index + match[0].length;
  }
  if (lastEnd < text.length) {
    spans.push([lastEnd, text.length]);
  }
  return spans;
}

function bisectRight(arr, val) {
  let lo = 0;
  let hi = arr.length;
  while (lo < hi) {
    const mid = Math.floor((lo + hi) / 2);
    if (arr[mid] <= val) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

function bisectLeft(arr, val) {
  let lo = 0;
  let hi = arr.length;
  while (lo < hi) {
    const mid = Math.floor((lo + hi) / 2);
    if (arr[mid] < val) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

function coveredSentenceRange(spans, start, end) {
  if (!spans.length || start >= end) return null;
  const starts = spans.map((s) => s[0]);
  const ends = spans.map((s) => s[1]);
  const lo = bisectRight(ends, start);
  const hi = bisectLeft(starts, end) - 1;
  if (lo >= spans.length || hi < 0 || lo > hi) return null;
  return [lo, hi];
}

function mergeIntervals(items) {
  if (!items.length) return [];
  const itemsSorted = items.slice().sort((a, b) => {
    if (a.lo !== b.lo) return a.lo - b.lo;
    if (a.hi !== b.hi) return a.hi - b.hi;
    return a.raw_start - b.raw_start;
  });
  const merged = [];
  let cur = { ...itemsSorted[0] };
  for (let i = 1; i < itemsSorted.length; i++) {
    const it = itemsSorted[i];
    if (it.lo <= cur.hi) {
      cur.hi = Math.max(cur.hi, it.hi);
      cur.raw_end = Math.max(cur.raw_end, it.raw_end);
    } else {
      merged.push(cur);
      cur = { ...it };
    }
  }
  merged.push(cur);
  return merged;
}

function tagStreamWithOffsets(text) {
  const tagged = tagger.tagSentence(text);
  const parts = [];
  const pieces = [];
  let streamPos = 0;
  let rawPos = 0;

  for (let i = 0; i < tagged.length; i++) {
    const token = tagged[i];
    const value = token.value;
    const tokenStart = text.indexOf(value, rawPos);
    if (tokenStart === -1) {
      rawPos += value.length;
      continue;
    }

    const out = token.pos && VERB_TAGS.has(token.pos) ? "VERB" : value;
    parts.push(out);
    pieces.push([streamPos, streamPos + out.length, tokenStart, tokenStart + value.length]);
    streamPos += out.length;
    rawPos = tokenStart + value.length;

    if (i < tagged.length - 1) {
      const nextToken = tagged[i + 1];
      const nextPos = text.indexOf(nextToken.value, rawPos);
      if (nextPos > rawPos) {
        const whitespace = text.substring(rawPos, nextPos);
        parts.push(whitespace);
        pieces.push([streamPos, streamPos + whitespace.length, rawPos, nextPos]);
        streamPos += whitespace.length;
        rawPos = nextPos;
      } else {
        parts.push(" ");
        pieces.push([streamPos, streamPos + 1, rawPos, rawPos]);
        streamPos += 1;
      }
    }
  }

  return { stream: parts.join(""), pieces };
}

function streamToRaw(pieces, ss, se) {
  const streamStarts = pieces.map((p) => p[0]);
  const streamEnds = pieces.map((p) => p[1]);
  const i = bisectRight(streamEnds, ss);
  const j = bisectLeft(streamStarts, se) - 1;
  if (i >= pieces.length || j < i) return null;
  const slice = pieces.slice(i, j + 1);
  return [Math.min(...slice.map((p) => p[2])), Math.max(...slice.map((p) => p[3]))];
}

function extractContrastMatches(text) {
  const tNorm = normalizeText(text);
  const spans = sentenceSpans(tNorm);
  const candidates = [];

  for (const [pname, pregex] of Object.entries(STAGE1_REGEXES)) {
    for (const match of tNorm.matchAll(pregex)) {
      const rs = match.index;
      const re = match.index + match[0].length;
      const rng = coveredSentenceRange(spans, rs, re);
      if (!rng) continue;
      const [lo, hi] = rng;
      candidates.push({
        lo,
        hi,
        raw_start: rs,
        raw_end: re,
        pattern_name: `S1_${pname}`,
        match_text: match[0].trim(),
      });
    }
  }

  const { stream, pieces } = tagStreamWithOffsets(tNorm);
  for (const [pname, pregex] of Object.entries(STAGE2_REGEXES)) {
    for (const match of stream.matchAll(pregex)) {
      const mapRes = streamToRaw(pieces, match.index, match.index + match[0].length);
      if (!mapRes) continue;
      const [rs, re] = mapRes;
      const rng = coveredSentenceRange(spans, rs, re);
      if (!rng) continue;
      const [lo, hi] = rng;
      candidates.push({
        lo,
        hi,
        raw_start: rs,
        raw_end: re,
        pattern_name: `S2_${pname}`,
        match_text: tNorm.substring(rs, re).trim(),
      });
    }
  }

  const merged = mergeIntervals(candidates);
  return merged.map((it) => {
    const blockStart = spans[it.lo][0];
    const blockEnd = spans[it.hi][1];
    return {
      pattern_name: it.pattern_name,
      sentence: tNorm.substring(blockStart, blockEnd).trim(),
      match_text: it.match_text,
      sentence_count: it.hi - it.lo + 1,
    };
  });
}

let input = "";
process.stdin.setEncoding("utf8");
process.stdin.on("data", (chunk) => {
  input += chunk;
});
process.stdin.on("end", () => {
  try {
    const payload = JSON.parse(input);
    const text = typeof payload.text === "string" ? payload.text : "";
    process.stdout.write(JSON.stringify({ matches: extractContrastMatches(text) }));
  } catch (err) {
    process.stderr.write(`${err && err.stack ? err.stack : err}\n`);
    process.exit(1);
  }
});
