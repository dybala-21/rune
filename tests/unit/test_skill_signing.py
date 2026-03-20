"""Tests for rune.skills.signing — ported from signing.test.ts."""


from rune.skills.signing import sign_skill_payload, verify_skill_signature


class TestSkillSigning:
    """Tests for sign/verify skill payloads."""

    def test_creates_verifiable_signature(self):
        payload = dict(
            name="signed-skill",
            description="desc",
            body="safe body",
            scope="user",
            author="rune-agent",
        )
        secret = "unit-test-secret"
        signature = sign_skill_payload(**payload, secret=secret)

        assert signature.startswith("hmac-sha256:")
        assert verify_skill_signature(**payload, secret=secret, signature=signature) is True

    def test_fails_verification_when_payload_differs(self):
        payload = dict(
            name="signed-skill",
            description="desc",
            body="safe body",
            scope="user",
            author="rune-agent",
        )
        secret = "unit-test-secret"
        signature = sign_skill_payload(**payload, secret=secret)

        tampered = {**payload, "body": "tampered body"}
        assert verify_skill_signature(**tampered, secret=secret, signature=signature) is False

    def test_fails_verification_with_wrong_secret(self):
        payload = dict(
            name="s",
            description="d",
            body="b",
            scope="user",
            author="a",
        )
        signature = sign_skill_payload(**payload, secret="correct-secret")
        assert verify_skill_signature(**payload, secret="wrong-secret", signature=signature) is False

    def test_fails_for_invalid_signature_format(self):
        payload = dict(
            name="s",
            description="d",
            body="b",
            scope="user",
            author="a",
        )
        assert verify_skill_signature(
            **payload, secret="s", signature="not-a-valid-signature"
        ) is False

    def test_signature_prefix_is_stripped_during_verification(self):
        payload = dict(
            name="test",
            description="desc",
            body="body",
            scope="user",
            author="me",
        )
        secret = "secret123"
        sig = sign_skill_payload(**payload, secret=secret)
        # Verify with the full prefixed signature
        assert verify_skill_signature(**payload, secret=secret, signature=sig) is True
        # Also verify with just the hex digest
        hex_digest = sig.removeprefix("hmac-sha256:")
        assert verify_skill_signature(**payload, secret=secret, signature=hex_digest) is True
