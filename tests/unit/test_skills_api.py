"""The skills API used to answer create/update/delete with hardcoded success.

The panel in the web UI called it, got ``{"deleted": true}``, and the skill was
still there. These tests pin the handlers to the real registry and to the guard
that keeps a delete inside the skills directory.
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from rune.api.handlers import skills as api
from rune.skills.registry import get_skill_registry
from rune.skills.types import Skill


@pytest.fixture
def skills_home(tmp_path, monkeypatch):
    home = tmp_path / "rune-home"
    home.mkdir()
    monkeypatch.setenv("RUNE_HOME", str(home))
    monkeypatch.chdir(tmp_path)
    api._reload_registry()
    yield home
    api._reload_registry()


async def _names() -> list[str]:
    return [s.name for s in (await api.list_skills()).skills]


class TestTheLifecycleIsReal:
    @pytest.mark.asyncio
    async def test_a_created_skill_is_on_disk_and_in_the_listing(self, skills_home):
        created = await api.create_skill(
            api.SkillCreateRequest(
                name="daily-briefing", description="d", body="Step one.", scope="user"
            )
        )

        assert (skills_home / "skills" / "daily-briefing" / "SKILL.md").is_file()
        assert await _names() == ["daily-briefing"]
        assert created.body == "Step one."

    @pytest.mark.asyncio
    async def test_an_update_reaches_the_file(self, skills_home):
        await api.create_skill(api.SkillCreateRequest(name="s-one", description="old"))

        updated = await api.update_skill(
            "s-one", api.SkillUpdateRequest(description="new", body="Body.")
        )

        assert updated.description == "new"
        assert "new" in (skills_home / "skills" / "s-one" / "SKILL.md").read_text()

    @pytest.mark.asyncio
    async def test_an_omitted_field_is_left_alone(self, skills_home):
        await api.create_skill(
            api.SkillCreateRequest(name="s-two", description="keep", body="Body.")
        )

        updated = await api.update_skill("s-two", api.SkillUpdateRequest(body="New."))

        assert updated.description == "keep"
        assert updated.body == "New."

    @pytest.mark.asyncio
    async def test_a_delete_removes_the_file_and_the_directory(self, skills_home):
        created = await api.create_skill(api.SkillCreateRequest(name="s-three"))
        skill_dir = skills_home / "skills" / "s-three"

        assert (await api.delete_skill("s-three")).deleted is True

        assert not skill_dir.exists()
        assert await _names() == []
        assert created.file_path


class TestBadInputIsRefusedRatherThanFaked:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("name", ["Bad Name", "", "../escape", "..", "a/b", "-x"])
    async def test_a_name_that_is_not_kebab_case_is_rejected(self, skills_home, name):
        with pytest.raises(HTTPException) as exc:
            await api.create_skill(api.SkillCreateRequest(name=name))
        assert exc.value.status_code == 400

    @pytest.mark.asyncio
    async def test_creating_the_same_name_twice_is_a_conflict(self, skills_home):
        await api.create_skill(api.SkillCreateRequest(name="dup"))
        with pytest.raises(HTTPException) as exc:
            await api.create_skill(api.SkillCreateRequest(name="dup"))
        assert exc.value.status_code == 409

    @pytest.mark.asyncio
    @pytest.mark.parametrize("call", ["get", "update", "delete"])
    async def test_an_unknown_skill_is_a_404(self, skills_home, call):
        with pytest.raises(HTTPException) as exc:
            if call == "get":
                await api.get_skill("ghost")
            elif call == "update":
                await api.update_skill("ghost", api.SkillUpdateRequest(body="x"))
            else:
                await api.delete_skill("ghost")
        assert exc.value.status_code == 404

    @pytest.mark.asyncio
    async def test_a_skill_outside_the_skills_directories_cannot_be_edited(
        self, skills_home
    ):
        """A programmatically registered skill has no file to rewrite."""
        get_skill_registry().register(
            Skill(name="builtin-one", description="d", scope="builtin")
        )

        with pytest.raises(HTTPException) as exc:
            await api.update_skill("builtin-one", api.SkillUpdateRequest(body="x"))

        assert exc.value.status_code == 409

class TestDeleteStaysInsideTheSkillsDirectory:
    @pytest.mark.asyncio
    async def test_a_path_outside_the_tree_is_refused(self, skills_home):
        victim = skills_home / "PRECIOUS.txt"
        victim.write_text("keep me")
        get_skill_registry().register(
            Skill(name="evil", description="d", scope="user", file_path=str(victim))
        )

        with pytest.raises(HTTPException) as exc:
            await api.delete_skill("evil")

        assert exc.value.status_code == 409
        assert victim.read_text() == "keep me"

    @pytest.mark.asyncio
    async def test_a_symlink_pointing_out_of_the_tree_is_refused(self, skills_home):
        victim = skills_home / "PRECIOUS.txt"
        victim.write_text("keep me")
        link_dir = skills_home / "skills" / "sneaky"
        link_dir.mkdir(parents=True)
        (link_dir / "SKILL.md").symlink_to(victim)
        get_skill_registry().register(
            Skill(
                name="sneaky",
                description="d",
                scope="user",
                file_path=str(link_dir / "SKILL.md"),
            )
        )

        with pytest.raises(HTTPException) as exc:
            await api.delete_skill("sneaky")

        assert exc.value.status_code == 409
        assert victim.is_file()

    @pytest.mark.asyncio
    async def test_a_registered_skill_with_no_file_is_refused(self, skills_home):
        get_skill_registry().register(
            Skill(name="in-memory", description="d", scope="user")
        )

        with pytest.raises(HTTPException) as exc:
            await api.delete_skill("in-memory")

        assert exc.value.status_code == 409


class TestFrontmatterCannotBeForged:
    """The registry reads frontmatter line by line, so a newline in a value
    used to start a new key — a description could rename the skill or set its
    scope to builtin, which made it undeletable through the API."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "description",
        [
            "innocent\nscope: builtin\nname: hijacked",
            "innocent\n---\nbody injected",
            "x\nfile_path: /etc/passwd",
            "x\r\nscope: builtin",
        ],
    )
    async def test_a_crafted_description_cannot_change_name_or_scope(
        self, skills_home, description
    ):
        created = await api.create_skill(
            api.SkillCreateRequest(name="probe", description=description, scope="user")
        )

        assert created.name == "probe"
        assert created.scope == "user"
        assert "\n" not in created.description
        assert (await api.delete_skill("probe")).deleted is True

    @pytest.mark.asyncio
    async def test_a_body_cannot_open_a_second_frontmatter(self, skills_home):
        created = await api.create_skill(
            api.SkillCreateRequest(
                name="body-probe",
                description="d",
                body="---\nscope: builtin\n---\nreal body",
                scope="user",
            )
        )

        assert created.scope == "user"
        assert created.name == "body-probe"

    @pytest.mark.asyncio
    async def test_a_skill_that_does_not_load_back_leaves_no_file(
        self, skills_home, monkeypatch
    ):
        """A half-written skill must not be waiting for the next registry scan."""
        monkeypatch.setattr(api, "_reload_registry", lambda: _EmptyRegistry())

        with pytest.raises(HTTPException) as exc:
            await api.create_skill(api.SkillCreateRequest(name="ghost", description="d"))

        assert exc.value.status_code == 500
        assert not (skills_home / "skills" / "ghost").exists()


class _EmptyRegistry:
    def get(self, name):  # noqa: ARG002
        return None


class TestScopeComesFromWhereTheFileIs:
    """Frontmatter is content the skill supplies, so it cannot be trusted for
    the two decisions that used to read it: whether the skill is editable, and
    which directory a delete is allowed to touch."""

    @pytest.mark.asyncio
    async def test_a_file_claiming_builtin_is_still_deletable(self, skills_home):
        skill_dir = skills_home / "skills" / "sneaky"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: sneaky\ndescription: d\nscope: builtin\n---\n\nB\n"
        )
        api._reload_registry()

        listed = await api.list_skills()
        assert next(s.scope for s in listed.skills if s.name == "sneaky") == "user"
        assert (await api.delete_skill("sneaky")).deleted is True

    @pytest.mark.asyncio
    async def test_a_project_file_claiming_user_scope_is_still_deletable(
        self, skills_home, tmp_path
    ):
        skill_dir = tmp_path / ".rune" / "skills" / "pskill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: pskill\ndescription: d\nscope: user\n---\n\nB\n"
        )
        api._reload_registry()

        listed = await api.list_skills()
        assert next(s.scope for s in listed.skills if s.name == "pskill") == "project"
        assert (await api.delete_skill("pskill")).deleted is True


class TestWritesFollowTheRealFile:
    @pytest.mark.asyncio
    async def test_create_refuses_to_overwrite_a_file_it_did_not_make(
        self, skills_home
    ):
        """The registry keys on frontmatter name, the path on directory name."""
        skill_dir = skills_home / "skills" / "notes"
        skill_dir.mkdir(parents=True)
        existing = skill_dir / "SKILL.md"
        existing.write_text(
            "---\nname: my-notes\ndescription: precious\n---\n\nIrreplaceable.\n"
        )
        api._reload_registry()

        with pytest.raises(HTTPException) as exc:
            await api.create_skill(api.SkillCreateRequest(name="notes"))

        assert exc.value.status_code == 409
        assert "precious" in existing.read_text()

    @pytest.mark.asyncio
    async def test_update_rewrites_the_file_the_skill_was_loaded_from(
        self, skills_home
    ):
        """A rebuilt path made a second file that lost to the original on reload."""
        alt = skills_home / "skills" / "notes2.skill.md"
        alt.parent.mkdir(parents=True)
        alt.write_text(
            "---\nname: notes2\ndescription: original\n---\n\nOriginal body.\n"
        )
        api._reload_registry()

        updated = await api.update_skill(
            "notes2", api.SkillUpdateRequest(description="edited")
        )

        assert updated.description == "edited"
        assert "edited" in alt.read_text()
        assert not (skills_home / "skills" / "notes2").exists()
        api._reload_registry()
        assert get_skill_registry().get("notes2").description == "edited"
