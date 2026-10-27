"""Ingredient-only candidate selection; no labels or filenames used at inference."""
import re

HEADER = re.compile(r'원\s*재\s*료|[재대]\s*료\s*명')
POSITIVE = re.compile(r'밀가루|정제수|설탕|백설탕|유크림|유당|식물성|팜유|전분|분유|농축|추출|향료|코코아|감자|유지|대두|구연산|산도조절|혼합|미국산|국내산|호주산|말레이시아산|인도네시아산|중국산|아스파탐|수크랄로스')
STOP = re.compile(r'영\s*양|소비기한|유통기한|품목|보고번호|포장|재질|제조원|판매원|소재지|고객|상담|반품|교환|보관|제조시설|혼입|함유|열량|나트륨\s*\d|탄수화물\s*\d')


def area(b):
    return max(0, b[2]-b[0])*max(0,b[3]-b[1])


def intersection(a,b):
    return area([max(a[0],b[0]),max(a[1],b[1]),min(a[2],b[2]),min(a[3],b[3])])


def selected_words(box, words):
    return [w for w in words if intersection(box,w[:4])/max(area(w[:4]),1) >= 0.5]


def evidence(box, words, penalty=2):
    rows = selected_words(box,words)
    text = ' '.join(w[4] for w in rows)
    positives = len(POSITIVE.findall(text))
    header = bool(HEADER.search(text))
    negatives = sum(bool(STOP.search(w[4])) and not HEADER.search(w[4]) for w in rows)
    return 3*header + min(positives,12) - penalty*negatives, positives + 2*header


def refine(box, words, shape, pad):
    """Trim vertical margins using ingredient evidence and following field headings."""
    rows = sorted(selected_words(box,words), key=lambda w:(w[1],w[0]))
    positive = [w for w in rows if HEADER.search(w[4]) or POSITIVE.search(w[4])]
    if not positive:
        return list(box)
    start = min(w[1] for w in positive)
    last = max(w[3] for w in positive)
    stops = [w[1] for w in rows if w[1] >= last-2 and STOP.search(w[4]) and not POSITIVE.search(w[4])]
    end = min(stops) if stops else box[3]
    # Never remove a positive row. Keep the proposal's horizontal extent.
    end = max(last,end)
    x1,y1,x2,y2 = box[0],max(box[1],start),box[2],min(box[3],end)
    margin = max(2,(y2-y1)*pad)
    return [max(0,x1-margin),max(0,y1-margin),min(shape[1],x2+margin),min(shape[0],y2+margin)]


def choose_region(baseline, proposals, words, shape, min_evidence=2, pad=0.02, penalty=2):
    candidates = [(refine(baseline,words,shape,pad),'rules_refined',0.0)]
    candidates += [(refine(p['bbox'],words,shape,pad),'model_verified',p['conf']) for p in proposals]
    ranked = []
    for box,source,confidence in candidates:
        score,support = evidence(box,words,penalty)
        if support >= min_evidence:
            ranked.append((score+0.2*confidence,-area(box),box,source))
    if not ranked:
        return list(baseline),'fallback_rules'
    chosen = max(ranked,key=lambda c:(c[0],c[1]))
    return chosen[2],chosen[3]
