# Rune for iOS

iPhone/iPad용 SwiftUI 앱입니다. 기존 Rune 웹 UI를 `WKWebView`로 표시하고 Mac 또는 서버의 Rune 엔진에 연결합니다. **Python 엔진·Playwright·macOS 앱 제어는 연결한 호스트에서 실행됩니다. iPhone 안에서 Rune 엔진이 독립 실행되거나 다른 iOS 앱을 직접 조작하는 구현은 아닙니다.**

## 구현 범위

- iOS 16 이상, iPhone/iPad, Xcode 프로젝트 및 공유 `Rune` scheme.
- HTTPS 서버 설정, 서버 내 탐색, 연결 오류 표시, 재연결, 연결 해제.
- 기존 채팅, 작업 이력, 승인, 서버 측 Computer 화면을 그대로 사용.
- Files의 텍스트 가져오기 → 미리보기 → 현재 채팅 초안에 추가. 기존 초안을 보존하고 자동 전송하지 않음.
- UTF-8 / BOM이 있는 UTF-16, 최대 256 KB 및 100,000 UTF-16 코드 단위. 큰 파일은 기존 채팅의 첨부 버튼 사용.
- 외부 웹 링크는 별도 Safari 화면에서 열기. Rune 로그인 쿠키와 분리.

토큰은 HTTPS 게이트웨이의 로그인 폼에서 입력합니다. 앱은 서버 주소만 UserDefaults에 저장합니다. 로그인 쿠키·웹 저장소는 메모리에 유지되고 앱 종료/연결 해제로 사라집니다. 서버에서 발급된 세션 자체는 기존 만료·폐기 정책을 따릅니다. 비밀번호·토큰을 URL, 자바스크립트 코드, UserDefaults에 주입하지 않습니다.

## Xcode에서 실행

1. Xcode와 기기에 맞는 iOS SDK를 설치합니다. Command Line Tools만으로는 앱을 빌드할 수 없습니다.
2. `ios/Rune.xcodeproj`를 엽니다.
3. `Rune` target → Signing & Capabilities에서 본인의 Team과 고유 Bundle Identifier를 지정합니다.
4. iPhone Simulator 또는 연결한 iPhone을 선택하고 Run 합니다. 실기기는 서명 및 필요 시 Developer Mode 설정이 필요합니다.
5. 아래에서 준비한 HTTPS 게이트웨이 주소를 입력하고 로그인합니다.

Xcode가 설치된 환경의 서명 없는 시뮬레이터 빌드:

```sh
xcodebuild -project ios/Rune.xcodeproj -scheme Rune \
  -configuration Debug -sdk iphonesimulator \
  -destination 'generic/platform=iOS Simulator' \
  -derivedDataPath ios/DerivedData CODE_SIGNING_ALLOWED=NO build
```

Debug **시뮬레이터**에서만 `http://localhost:18789`를 허용합니다. 이 경우 일반 `uv run rune web --no-open`에 연결하며, `RUNE_REQUIRE_TOKEN=1`이 설정된 서버에는 직접 연결하지 않습니다. 실제 iPhone의 localhost는 Mac이 아닙니다. 실기기와 Release 빌드는 HTTPS를 사용합니다. `NSAllowsLocalNetworking`은 로컬 네트워크 용도이며 전체 ATS 해제나 인증서 검증 우회는 없습니다.

## 기존 Mac 엔진을 HTTPS로 연결

새 클라우드 VM 없이 기존 인증 게이트웨이를 단일 사용자용으로 재사용할 수 있습니다. 아래 명령은 저장소 루트에서 실행합니다. 게이트웨이는 연결된 Rune 계정의 작업·파일 권한을 제공하므로 본인의 서버에 연결하세요.

프런트엔드를 빌드하고 서버를 시작합니다. 이미 실행 중인 Rune이 있다면 동일 포트에서 중복 실행하지 않습니다.

```sh
npm --prefix web ci
npm --prefix web run build
RUNE_REQUIRE_TOKEN=1 uv run rune web --no-open --host 127.0.0.1 --port 18789
```

다른 터미널에서 같은 `RUNE_HOME` 설정으로 내부 토큰을 생성합니다. 이 파일은 게이트웨이와 엔진 사이의 토큰이며, iPhone에 입력할 토큰과 다릅니다. 파일이 이미 존재하면 덮어쓰지 않습니다.

```sh
uv run python - <<'PY'
import os
from rune.api.auth import generate_token
from rune.utils.paths import rune_home

path = rune_home() / 'ios-worker-token'
path.parent.mkdir(parents=True, exist_ok=True)
fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(fd, 'w') as file:
    file.write(generate_token(label='iOS gateway'))
print(f'Internal token saved to {path}')
PY
```

기본 `RUNE_HOME` 기준으로 등록합니다. 별도 경로를 사용하는 경우 방금 출력된 파일 경로를 사용합니다. **등록 시 출력되는 Access token**이 iPhone 로그인용입니다. 같은 owner를 재등록하면 기존 로그인과 세션이 교체됩니다.

```sh
uv run rune cloud register ios-personal http://127.0.0.1:18789 ~/.rune/ios-worker-token
uv run rune cloud serve https://rune.example.com --host 127.0.0.1 --port 18800
```

`https://rune.example.com`은 소유한 실제 도메인으로 교체합니다. 해당 HTTPS 엔드포인트의 TLS reverse proxy를 `127.0.0.1:18800`에 연결해야 합니다. 프록시는 원래 경로, SSE 스트리밍과 WebSocket upgrade를 전달해야 합니다. `18789`와 `18800`을 직접 인터넷에 공개하지 않습니다. iPhone에서 신뢰하는 인증서가 필요하며, 앱과 gateway의 `public_origin`은 scheme·host·port가 일치해야 합니다. 이 저장소 변경은 도메인/TLS 배포를 자동으로 만들지 않습니다.

접근을 폐기하려면:

```sh
uv run rune cloud disable ios-personal
```

## 카카오톡 대화 요약 사용 흐름

1. 사용자가 선택한 대화를 카카오톡에서 텍스트로 내보내 Files에 저장합니다. 내보내기 메뉴와 지원 범위는 설치된 카카오톡 버전에서 확인합니다.
2. Rune 서버에 연결하고 채팅을 엽니다.
3. 상단 메뉴 → **Import text** → 파일 선택 → 미리보기 → **Add to chat draft**.
4. 원하는 요약 조건을 입력하고 전송합니다. 예: “이 대화에서 결정 사항, 할 일, 담당자, 날짜를 정리해줘.”

파일 미리보기는 기기에서 처리됩니다. 초안에 추가하면 연결된 Rune 웹 화면의 코드가 내용을 읽을 수 있으므로 신뢰하는 서버만 사용합니다. 이 구현은 Send 전 자동 업로드를 하지 않습니다. 전송 후에는 서버의 대화 저장 정책과 선택한 모델 제공자의 데이터 처리 범위가 적용됩니다. 카카오톡 DB·알림·다른 앱의 화면을 몰래 읽는 기능은 없습니다.

## 검증과 남은 범위

이 개발 환경에는 Xcode/iOS SDK가 없어 **실제 iOS 앱 빌드, 시뮬레이터 실행, 실기기 설치는 미검증**입니다. 현재 검증은 다음 범위입니다.

- Swift Foundation 기반 URL 경계·파일 인코딩·크기 제한: 29 checks 통과.
- Swift 소스 구문 검사 및 Xcode 프로젝트/Info/Privacy plist 파싱 통과. UIKit/SwiftUI/WebKit의 iOS 타입 검사는 포함하지 않음.
- 웹 테스트 114개 통과, TypeScript 및 프로덕션 빌드 통과.
- 기존 인증 gateway 테스트 10개 통과. 네이티브 WebKit과의 종단 간 로그인 검증은 별도 필요.
- Chromium 모바일 smoke 통과: 390×760, 844×346, 768×960 화면에서 입력창 노출, 기존 초안 보존, Unicode/코드 문자열의 문자 그대로 가져오기, 크기 초과 거부, 자동 전송 없음. 화면 캡처도 확인.

재현 명령:

```sh
mkdir -p /tmp/rune-ios-module-cache
swiftc -module-cache-path /tmp/rune-ios-module-cache \
  ios/Rune/Core/ServerAddress.swift ios/Rune/Core/TextImport.swift \
  ios/Tests/main.swift -o /tmp/rune-ios-core-tests
/tmp/rune-ios-core-tests
npm --prefix web test
npm --prefix web run build
uv run pytest tests/unit/test_cloud_gateway.py -q
```

모바일 브라우저 스모크 테스트(엔진·SSE fixture, 실제 모델 호출 없음):

```sh
npm --prefix web run preview -- --host 127.0.0.1 --port 4179 --strictPort
# 다른 터미널
uv run python ios/Tests/mobile_smoke.py
```

이는 Chromium 모바일 화면 검사이며 WKWebView 검사를 대체하지 않습니다. 실기기에서 로그인/8시간 세션 만료 후 재로그인, 한글 키보드, 파일 선택, 외부 링크, 백그라운드 복귀, SSE 재연결, 승인 응답, 서버 Computer 조작을 추가 확인해야 합니다. 앱을 백그라운드로 보내면 UI 연결은 중단될 수 있지만 서버 작업은 계속될 수 있습니다. 재연결은 루트 GET만 수행하며 작업 POST를 다시 보내지 않습니다. 미전송 초안은 재연결/프로세스 종료로 사라질 수 있습니다.

공유 확장, App Intents, 푸시 알림, 오프라인 엔진, iPhone 앱 제어 driver, 결제·예약 connector는 포함하지 않습니다. TestFlight/App Store 업로드도 수행하지 않았습니다. 배포 전 서비스의 실제 데이터 처리에 맞는 개인정보 고지·App Privacy 응답과 심사 요건을 별도로 검토해야 합니다.

앱 조작 가능성, 공식 근거 및 다음 실험은 [리서치 문서](RESEARCH.ko.md)를 참고하세요.
