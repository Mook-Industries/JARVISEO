from hybrid_region import choose_region, refine, area

def test_no_semantic_support_falls_back():
    baseline=[0,0,100,100]
    result,source=choose_region(baseline,[{'bbox':[5,5,50,50],'conf':0.99}],[[5,5,30,20,'제품 광고']],(100,100,3))
    assert result==baseline and source=='fallback_rules'

def test_high_confidence_nutrition_is_not_ingredient_evidence():
    result,source=choose_region([0,0,100,100],[{'bbox':[0,0,50,30],'conf':0.99}],
                               [[0,0,50,30,'영양정보 나트륨 100mg']],(100,100,3))
    assert source=='fallback_rules'

def test_semantic_candidate_preserves_positive_text_and_stays_in_bounds():
    words=[[10,20,90,35,'원재료명 밀가루 설탕'],[10,40,90,55,'팜유 코코아'],[10,60,90,75,'품목보고번호 123']]
    b=refine([0,0,100,100],words,(100,100,3),0.02)
    assert b[1]<=20 and b[3]>=55 and b[3]<75
    assert 0<=b[0]<b[2]<=100 and 0<=b[1]<b[3]<=100 and area(b)>0
