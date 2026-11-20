from tsarchive.heads import Head, associate, people_count, person_codes
from tsarchive.helmet import H_NO, H_UNKNOWN, H_YES


def test_two_heads_in_one_person_box_count_as_two_people():
    persons = [[0.35, 0.28, 0.62, 0.94], [0.61, 0.35, 0.75, 0.96]]  # общая рамка двоих + отдельный человек
    heads = [Head([0.42, 0.30, 0.50, 0.42], 0.9, H_YES), Head([0.50, 0.33, 0.58, 0.45], 0.88, H_YES),
             Head([0.63, 0.38, 0.70, 0.50], 0.8, H_NO)]
    owned = associate(persons, heads)
    assert owned == [[0, 1], [2]]
    n, extra = people_count(owned, heads, 0.6)
    assert n == 3 and extra == [1]
    assert person_codes(owned, heads) == [(H_YES, 0.0), (H_NO, 0.8)]


def test_head_without_person_box():
    heads = [Head([0.1, 0.1, 0.15, 0.18], 0.7, H_NO), Head([0.8, 0.1, 0.85, 0.18], 0.4, H_YES)]
    owned = associate([], heads)
    n, extra = people_count(owned, heads, 0.6)
    assert n == 1 and extra == [0]


def test_head_below_box_top_not_matched():
    persons = [[0.2, 0.1, 0.4, 0.9]]
    heads = [Head([0.25, 0.7, 0.3, 0.76], 0.9, H_YES)]  # «голова» у ног — не этого человека
    owned = associate(persons, heads)
    assert owned == [[]]
    assert person_codes(owned, heads) == [(H_UNKNOWN, 0.0)]
