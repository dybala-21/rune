"""Validate the schema that reaches the provider, including nested references."""

from copy import deepcopy

from jsonschema import Draft202012Validator

from rune.agent.litellm_adapter import tools_to_openai_schema
from rune.agent.tool_adapter import ToolAdapterOptions, build_tool_set
from rune.capabilities.document_bundle import register_document_bundle_capability
from rune.capabilities.registry import CapabilityRegistry


def test_nested_bundle_contract_reaches_model_and_validates_partial_patch(monkeypatch):
    monkeypatch.setenv('RUNE_BUNDLE_UPDATE_ENABLED', '1')
    registry = CapabilityRegistry()
    register_document_bundle_capability(registry)
    tools = build_tool_set(ToolAdapterOptions(enable_guardian=False), registry=registry)
    tool = tools['document_bundle_update']
    wire = tools_to_openai_schema([tool])[0]['function']['parameters']
    validator = Draft202012Validator(wire)
    validator.validate({'directory': 'bundle', 'base_revision': 'a' * 32,
                        'expected_source_sha256': 'b' * 64,
                        'changes': {'documents': [{'filename': 'briefing.docx', 'title': 'New title'}]}})
    invalid = {'directory': 'bundle', 'base_revision': 'a' * 32,
               'expected_source_sha256': 'b' * 64,
               'changes': {'filters': [{'column': 'status', 'operator': 'not-a-real-operator'}]}}
    assert list(validator.iter_errors(invalid))
    create = tools['document_bundle'].json_schema
    assert set(create['$defs']['BundleMetric']['required']) == {'id', 'operation'}
    assert create['additionalProperties'] is False


def test_raw_mcp_schema_is_preserved_without_mutating_registration():
    from rune.capabilities.types import CapabilityDefinition
    from rune.types import Domain, RiskLevel

    raw = {'type': 'object', 'additionalProperties': False,
           '$defs': {'Entry': {'type': 'string', 'minLength': 2}},
           'properties': {'value': {'$ref': '#/$defs/Entry'}}, 'required': ['value']}
    before = deepcopy(raw)
    registry = CapabilityRegistry()
    registry.register(CapabilityDefinition(name='custom_tool', description='test', domain=Domain.FILE,
                                           risk_level=RiskLevel.LOW, raw_json_schema=raw))
    tools = build_tool_set(ToolAdapterOptions(enable_guardian=False), registry=registry)
    wire = tools_to_openai_schema(list(tools.values()))[0]['function']['parameters']
    validator = Draft202012Validator(wire)
    validator.validate({'value': 'ok'})
    for invalid in ({}, {'value': 'x'}, {'value': 3}, {'value': 'ok', 'extra': True}):
        assert list(validator.iter_errors(invalid)), invalid
    assert raw == before
