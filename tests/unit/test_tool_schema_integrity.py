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


def test_compaction_preserves_title_fields_values_and_validation():
    from rune.agent.tool_adapter import ToolWrapper

    schema = {'type': 'object', 'title': 'Document', 'additionalProperties': False,
              'properties': {
                  'title': {'type': 'string', 'title': 'Title', 'minLength': 1},
                  'recipient': {'type': 'string', 'title': 'Account owner'},
                  'metadata': {'type': 'object', 'default': {'title': 'Saved title'},
                               'examples': [{'title': 'Example title'}]},
                  'items': {'type': 'array', 'title': 'Items', 'items': {'$ref': '#/$defs/Entry'}}},
              '$defs': {'Entry': {'title': 'Entry', 'type': 'integer', 'minimum': 1}},
              'required': ['title', 'items']}
    original = deepcopy(schema)
    tool = ToolWrapper(name='document', json_schema=schema, description='Create a document')
    compact = tools_to_openai_schema([tool])[0]['function']['parameters']
    assert schema == original and 'think' not in schema['properties']
    assert compact['properties']['title'] == {'type': 'string', 'minLength': 1}
    assert compact['properties']['recipient']['title'] == 'Account owner'
    assert compact['properties']['metadata'] == schema['properties']['metadata']
    assert 'title' not in compact['$defs']['Entry']
    for value in ({'title': 'Report', 'items': [1]}, {'items': []}, {'title': '', 'items': [0]},
                  {'title': 'Report', 'items': [1], 'extra': True}):
        assert Draft202012Validator(compact).is_valid(value) == Draft202012Validator(schema).is_valid(value)


def test_native_reasoning_omits_only_the_optional_harness_rationale():
    from rune.agent.tool_schema import CALL_RATIONALE, without_call_rationale

    def tool(properties, required=()):
        return {'type': 'function', 'function': {'name': 'inspect', 'parameters': {
            'type': 'object', 'properties': properties, 'required': list(required)}}}
    schemas = [tool({'path': {'type': 'string'}, 'think': dict(CALL_RATIONALE)}),
               tool({'think': {'type': 'integer'}}, ['think']),
               tool({'think': dict(CALL_RATIONALE)}, ['think']), {'type': 'advisor'}]
    original = deepcopy(schemas)
    wire = without_call_rationale(schemas)
    assert schemas == original
    assert wire[0]['function']['parameters']['properties'] == {'path': {'type': 'string'}}
    assert wire[1:] == schemas[1:]
