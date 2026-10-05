"""GS1 prefix ranges, the fifth constraint.

The first three digits of an EAN-13 are assigned by GS1 to a member
organisation (usually a country), to restricted circulation, to coupons or to
books. A reconstruction that lands in an unassigned range is not a product
anyone could have printed, so it is discarded.

Transcribed 2026-10-05 from en.wikipedia.org/wiki/List_of_GS1_country_codes,
because gs1.org refuses scripted fetches. Gaps are computed from this list, not
typed by hand. Re-check against gs1.org before the report quotes it.

GTIN-8 ranges (960-969) are deliberately absent: those prefixes belong to
8-digit codes and do not start a legitimate EAN-13.
"""

ASSIGNED = [
    (0, 19, "GS1 US"),
    (20, 29, "restricted circulation, geographic"),
    (30, 39, "GS1 US, drugs (NDC)"),
    (40, 49, "restricted circulation, company"),
    (50, 59, "GS1 US, reserved / coupons"),
    (60, 139, "GS1 US"),
    (200, 299, "restricted circulation, geographic"),
    (300, 379, "France and Monaco"),
    (380, 380, "Bulgaria"), (381, 381, "Kosovo"), (383, 383, "Slovenia"),
    (385, 385, "Croatia"), (387, 387, "Bosnia and Herzegovina"),
    (389, 389, "Montenegro"),
    (400, 440, "Germany"),
    (450, 459, "Japan"), (460, 469, "Russia"),
    (470, 470, "Kyrgyzstan"), (471, 471, "Taiwan"), (474, 474, "Estonia"),
    (475, 475, "Latvia"), (476, 476, "Azerbaijan"), (477, 477, "Lithuania"),
    (478, 478, "Uzbekistan"), (479, 479, "Sri Lanka"),
    (480, 480, "Philippines"), (481, 481, "Belarus"), (482, 482, "Ukraine"),
    (483, 483, "Turkmenistan"), (484, 484, "Moldova"), (485, 485, "Armenia"),
    (486, 486, "Georgia"), (487, 487, "Kazakhstan"), (488, 488, "Tajikistan"),
    (489, 489, "Hong Kong"),
    (490, 499, "Japan"),
    (500, 509, "United Kingdom"),
    (520, 521, "Greece"), (528, 528, "Lebanon"), (529, 529, "Cyprus"),
    (530, 530, "Albania"), (531, 531, "North Macedonia"), (535, 535, "Malta"),
    (539, 539, "Ireland"), (540, 549, "Belgium and Luxembourg"),
    (560, 560, "Portugal"), (569, 569, "Iceland"), (570, 579, "Denmark"),
    (590, 590, "Poland"), (594, 594, "Romania"), (599, 599, "Hungary"),
    (600, 601, "South Africa"),
    (603, 603, "Ghana"), (604, 604, "Senegal"), (605, 605, "Uganda"),
    (606, 606, "Angola"), (607, 607, "Oman"), (608, 608, "Bahrain"),
    (609, 609, "Mauritius"), (611, 611, "Morocco"), (612, 612, "Somalia"),
    (613, 613, "Algeria"), (615, 615, "Nigeria"), (616, 616, "Kenya"),
    (617, 617, "Cameroon"), (618, 618, "Ivory Coast"), (619, 619, "Tunisia"),
    (620, 620, "Tanzania"), (621, 621, "Syria"), (622, 622, "Egypt"),
    (623, 623, "GS1 Global Office"), (624, 624, "Libya"),
    (625, 625, "Jordan"), (626, 626, "Iran"), (627, 627, "Kuwait"),
    (628, 628, "Saudi Arabia"), (629, 629, "United Arab Emirates"),
    (630, 630, "Qatar"), (631, 631, "Namibia"), (632, 632, "Rwanda"),
    (640, 649, "Finland"),
    (680, 681, "China"), (690, 699, "China"),
    (700, 709, "Norway"), (729, 729, "Israel"), (730, 739, "Sweden"),
    (740, 740, "Guatemala"), (741, 741, "El Salvador"),
    (742, 742, "Honduras"), (743, 743, "Nicaragua"),
    (744, 744, "Costa Rica"), (745, 745, "Panama"),
    (746, 746, "Dominican Republic"), (750, 750, "Mexico"),
    (754, 755, "Canada"), (759, 759, "Venezuela"),
    (760, 769, "Switzerland and Liechtenstein"), (770, 771, "Colombia"),
    (773, 773, "Uruguay"), (775, 775, "Peru"), (777, 777, "Bolivia"),
    (778, 779, "Argentina"), (780, 780, "Chile"), (784, 784, "Paraguay"),
    (786, 786, "Ecuador"), (789, 790, "Brazil"),
    (800, 839, "Italy"), (840, 849, "Spain and Andorra"), (850, 850, "Cuba"),
    (858, 858, "Slovakia"), (859, 859, "Czech Republic"),
    (860, 860, "Serbia"), (865, 865, "Mongolia"), (867, 867, "North Korea"),
    (868, 869, "Turkey"), (870, 879, "Netherlands"),
    (880, 881, "South Korea"), (883, 883, "Myanmar"),
    (884, 884, "Cambodia"), (885, 885, "Thailand"), (887, 887, "Laos"),
    (888, 888, "Singapore"), (890, 890, "India"), (893, 893, "Vietnam"),
    (894, 894, "Bangladesh"), (896, 896, "Pakistan"),
    (899, 899, "Indonesia"),
    (900, 919, "Austria"), (930, 939, "Australia"),
    (940, 949, "New Zealand"),
    (950, 950, "GS1 Global Office"), (951, 951, "EPC general manager"),
    (952, 952, "demonstrations and examples"),
    (955, 955, "Malaysia"), (958, 958, "Macau"),
    (977, 977, "ISSN"), (978, 979, "ISBN / ISMN"),
    (980, 980, "refund receipts"), (981, 983, "coupons, common currency"),
    (990, 999, "coupons"),
]

_TABLE = [None] * 1000
for _lo, _hi, _name in ASSIGNED:
    for _p in range(_lo, _hi + 1):
        _TABLE[_p] = _name


def assignment(code):
    """Who the prefix belongs to, or None if nobody does."""
    return _TABLE[int(code[:3])]


def is_assigned(code):
    return _TABLE[int(code[:3])] is not None


def unassigned_prefixes():
    return [p for p in range(1000) if _TABLE[p] is None]
