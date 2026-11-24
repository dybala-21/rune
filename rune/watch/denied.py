"""Sites RUNE does not watch because their own terms forbid automated checking.

robots.txt says what a site lets a program read; a site's terms can forbid it apart from that, and
most shops' terms do. A site is listed here only once the rule's own words have been read where
the site publishes them: each entry gives the page, the version, the day it was read and the words
themselves. A summary of a site's terms, a model's or anybody's, does not list it, and a site that
is not listed has not been found to allow anything.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Rule:
    where: str    # the page the rule is written on
    version: str  # the day the version read came into force, or "undated" where the page gives none
    read: str     # the day it was read
    quote: str    # the rule, in the page's own words
    says: str     # what that forbids and to whom, what it leaves open, and how it was read


_MUSINSA = Rule(
    where="https://member.one.musinsa.com/agreement/service",
    version="2026-10-28",
    read="2026-11-23",
    quote=("회사와 사전 협의 또는 동의 없이 데이터 등을 추출하기 위하여 에이전트(Agent), 로봇(Robot), 크롤러(Crawler), "
           "스크립트(Script), 스파이더(Spider), 스파이웨어(Spyware), 매크로 프로그램 등의 자동화된 수단이나 수동 프로세스를 "
           "사용하여 몰 등 회사가 제공하는 서비스 및 서버에 접속하거나 몰 등 회사가 제공하는 서비스 및 서버에 포함된 콘텐츠 및 "
           "정보를 복사, 수집하거나 모니터링하는 행위"),
    says=("Article 11 (2) 9: without the company's agreement, no user may reach the mall or its servers, or copy, "
          "collect or monitor what is on them, by an agent, robot, crawler, script or other automated means. "
          "Article 2 counts members and non-members alike as users, and the Musinsa, 29CM and Soldout sites as "
          "the mall. Read in the Internet Archive's copy of 2026-11-15, which holds the page's text."),
)

# A site by the name it is registered under: every host under that name is the same site here.
DENIED: dict[str, Rule] = {
    "musinsa.com": _MUSINSA,
    "29cm.co.kr": _MUSINSA,
    "soldout.co.kr": _MUSINSA,
    "naver.com": Rule(
        where="https://policy.naver.com/policy/disclaimer.html",
        version="undated",
        read="2026-11-23",
        quote=("네이버는 아래의 경우를 제외하고, 자동화된 수단 및 방식(예: 매크로 프로그램, 로봇[봇], 크롤러, 스파이더, 스크래퍼 "
               "등)으로 네이버 서비스 또는 네이버의 API 서버 등에 접속하거나 정보를 수집하는 행위를 금지합니다."),
        says=("Legal notice, on reaching and using the service: nobody may reach Naver's services or collect from "
              "them by automated means, except with Naver's leave, through its API under a client id, or within "
              "what its robots.txt allows. The robots.txt of its shop hosts (shopping, smartstore, brand) allows "
              "a program it does not name nothing, read the same day, so no price page is left open. Its terms "
              "of use in force since 2026-10-01 speak to members and non-members."),
    ),
}


def denied(host: str) -> bool:
    """Whether ``host`` is a listed site or a host under one, however the name is spelled: the
    browser's way out is asked about a name as the browser wrote it."""
    name = host.rstrip(".").lower()
    return any(name == site or name.endswith("." + site) for site in DENIED)
