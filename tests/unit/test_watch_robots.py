"""robots.txt read the RFC 9309 way: the group for RUNE, the longest rule, wildcards, and refusals."""

import pytest

from rune.watch.robots import Rules, allows, parse, read

# RFC 9309 section 5.2, and the pair every first-match parser gets wrong.
STANDARD = """\
User-agent: *
Disallow: /example/page/disallowed.gif
Allow: /example/page/
Disallow: *.gif$
Disallow: /admin/
Allow: /admin/public/

User-agent: ExampleBot
Disallow: /
"""


@pytest.mark.parametrize(("path", "allowed"), [
    ("/example/page/a.html", True), ("/example/page/disallowed.gif", False), ("/a/b.gif", False),
    ("/a/b.gif?x=1", True), ("/admin/x", False), ("/admin/public/x", True), ("/", True), ("/robots.txt", True),
])
def test_the_longest_matching_rule_decides(path, allowed):
    assert allows(parse(STANDARD), "https://shop.example" + path) is allowed


def test_a_group_that_names_rune_is_obeyed_instead_of_the_one_for_everyone():
    text = "User-agent: *\nDisallow: /\n\nUser-agent: rune\nAllow: /products/\nDisallow: /\n"
    assert allows(parse(text), "https://shop.example/products/1")
    assert not allows(parse(text), "https://shop.example/cart")
    # Shops that open their product pages to assistants name them: a name RUNE doesn't carry is not RUNE's.
    named = "User-agent: *\nDisallow: /\n\nUser-agent: ChatGPT-User\nUser-agent: Claude-User\nAllow: /products/\n"
    assert not allows(parse(named), "https://shop.example/products/1")
    assert allows(parse(named, "claude-user"), "https://shop.example/products/1")


def test_rules_say_whether_the_site_wrote_them_for_rune_by_name():
    assert not parse("User-agent: *\nDisallow: /\n").named and not parse("").named
    assert parse("User-agent: *\nDisallow: /\n\nUser-agent: rune\nAllow: /p/\n").named
    assert not parse("User-agent: RuneScapeBot\nDisallow: /\n").named, "another program's name that begins the same"


def test_groups_for_the_same_name_are_read_together():
    text = "User-agent: a\nUser-agent: b\nDisallow: /p\n\nUser-agent: a\nAllow: /p/ok\n"
    assert allows(parse(text, "a"), "https://x.example/p/ok")
    assert not allows(parse(text, "b"), "https://x.example/p/ok")


def test_wildcards_as_real_shops_write_them():
    apple = parse("User-agent: *\nDisallow: /*shop/browse/overlay/*\nDisallow: /*shop/iphone/payments/overlay/*\n")
    assert allows(apple, "https://www.apple.com/kr/shop/buy-airpods/airpods-pro-3")
    assert not allows(apple, "https://www.apple.com/kr/shop/browse/overlay/x")
    eleven = parse("User-agent: *\nDisallow: /\nUser-agent: Yeti\nCrawl-delay: 1\nAllow: /*.gif$\nAllow: /products/\n")
    assert not allows(eleven, "https://www.11st.co.kr/products/1234567890")


def test_a_byte_order_mark_does_not_hide_the_first_group():
    # As Gmarket's item host serves it. Read with the mark kept, "everyone" is lost and RUNE looks allowed.
    text = "\ufeffUser-agent: *\r\nDisallow: /\r\n \r\nUser-agent: Googlebot\r\nAllow: /\r\n"
    assert not allows(parse(text), "https://item.gmarket.co.kr/Item?goodscode=1234567890")
    assert allows(parse(text, "googlebot"), "https://item.gmarket.co.kr/Item?goodscode=1234567890")


@pytest.mark.parametrize(("rule", "url", "allowed"), [
    ("/%ea%b2%80", "https://x.example/%EA%B2%80", False),      # the same escape in another case
    ("/검색", "https://x.example/%EA%B2%80%EC%83%89?q=1", False),  # written out against escaped
    ("/p%72oducts/", "https://x.example/products/1", False),   # a plain letter, escaped
    ("/a%2Fb", "https://x.example/a/b", True),                 # an escaped slash is not a slash
])
def test_escapes_are_compared_in_one_spelling(rule, url, allowed):
    assert allows(parse(f"User-agent: *\nDisallow: {rule}\n"), url) is allowed


def test_a_rule_is_matched_from_the_start_of_the_path_and_robots_txt_itself_is_always_readable():
    rules = parse("User-agent: *\nDisallow: /private\n")
    assert not allows(rules, "https://x.example/private/a") and allows(rules, "https://x.example/a/private")
    closed = parse("User-agent: *\nDisallow: /\n")
    assert not allows(closed, "https://x.example/") and allows(closed, "https://x.example/robots.txt")


def test_an_empty_rule_restricts_nothing_and_other_records_do_not_break_a_group():
    assert allows(parse("User-agent: *\nDisallow:\n"), "https://x.example/anything")
    text = "User-agent: *\nSitemap: https://x.example/sitemap.xml\nCrawl-delay: 3\nDisallow: /private\n"
    rules = parse(text)
    assert not allows(rules, "https://x.example/private/a") and rules.crawl_delay == 3.0
    assert parse("User-agent: *\nCrawl-delay: soon\nDisallow: /a\n").crawl_delay == 0.0
    # Only a rule ends a list of names (RFC 9309 2.2.1): a pause written between two names is for both.
    both = parse("User-agent: bot\nCrawl-delay: 9\n\nUser-agent: *\nDisallow: /a\n")
    assert both.crawl_delay == 9.0 and not allows(both, "https://x.example/a")
    after_a_rule = parse("User-agent: bot\nDisallow: /b\nCrawl-delay: 9\n\nUser-agent: *\nDisallow: /a\n")
    assert after_a_rule.crawl_delay == 0.0 and allows(after_a_rule, "https://x.example/b")


@pytest.mark.parametrize(("status", "expected"), [
    (200, Rules((((False, "/")),))), (404, Rules()), (410, Rules()),
    (401, "blocked"), (403, "blocked"), (429, "blocked"), (500, "robots"), (503, "robots"),
])
def test_what_a_robots_response_means(status, expected):
    assert read(status, b"User-agent: *\nDisallow: /\n") == expected


def test_only_the_first_part_of_a_very_long_file_is_read():
    filler = b"# padding\n" * 60_000
    assert not allows(read(200, b"User-agent: *\nDisallow: /a\n" + filler), "https://x.example/a")
    late = read(200, filler + b"User-agent: *\nDisallow: /a\n")
    assert isinstance(late, Rules) and allows(late, "https://x.example/a")
