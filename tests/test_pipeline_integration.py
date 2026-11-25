"""Exercise actual agent loop, tools, graph, renderer and review with a fake VLM."""
from types import SimpleNamespace

import pytest
from PIL import Image

from src.core.orchestrator import run_pipeline
from src.core.tool import ToolRegistry
from src.core.types import AgentRole, EventType, StreamEvent
from src.tools.visual.crop_roi import CropROITool
from src.tools.visual.link_evidence import LinkEvidenceTool
from src.tools.report.build_finding import BuildFindingTool
from src.tools.report.build_impression import BuildImpressionTool
from src.tools.review.claim_review import RecordClaimReviewTool


@pytest.mark.asyncio
async def test_complete_pipeline_binds_real_tools_and_preserves_original_producer(monkeypatch, tmp_path):
    holder = {}

    def build(role, settings, state):
        holder['state'] = state
        registry = ToolRegistry()
        for tool in {
            AgentRole.VISUAL_ANALYSIS: [CropROITool(), LinkEvidenceTool()],
            AgentRole.EVIDENCE_RETRIEVAL: [],
            AgentRole.REPORT_WRITING: [BuildFindingTool(), BuildImpressionTool()],
            AgentRole.QUALITY_REVIEW: [RecordClaimReviewTool()],
        }[role]:
            registry.register(tool)
        return SimpleNamespace(system_prompt=role.value, tool_registry=registry, hook_engine=None,
                               max_turns=8, max_tokens=8192, temperature=.7)

    class VLM:
        def __init__(self):
            self.turns = {}

        async def stream(self, **kwargs):
            stage = kwargs['system_prompt']
            turn = self.turns.get(stage, 0)
            self.turns[stage] = turn + 1
            state = holder['state']
            name, args = None, {}
            if stage == 'visual_analysis' and turn == 0:
                name, args = 'crop_roi', {'x': 0, 'y': 0, 'width': 4, 'height': 4, 'description': 'Synthetic region'}
            elif stage == 'visual_analysis' and turn == 1:
                image_id = state.active_image_id
                # crop is explicit in the previous tool reply; choose its actual ID.
                import json
                replies = [m for m in kwargs['messages'] if m['role'] == 'tool']
                image_id = json.loads(replies[-1]['content'])['image_id']
                name, args = 'link_visual_evidence', {'finding_description': 'No focal opacity.',
                    'image_evidence_ids': [image_id], 'confidence': .8, 'anatomy': 'lung', 'polarity': 'negative'}
            elif stage == 'report_writing' and turn == 0:
                name, args = 'build_structured_finding', {'text': 'No focal opacity.', 'anatomy': 'lung',
                    'type': 'normal', 'claim_type': 'normality', 'polarity': 'negative',
                    'image_evidence_ids': [state.image_evidence[0].image_id],
                    'observation_ids': [state.observation_graph[0]['observation_id']]}
            elif stage == 'report_writing' and turn == 1:
                name, args = 'build_structured_impression', {'text': 'No focal opacity.',
                    'linked_finding_ids': [state.structured_report.findings[0].finding_id]}
            elif stage == 'quality_review' and turn == 0:
                from src.report.grounding import effective_support, claim_id
                reviews = []
                for claim in state.structured_report.findings + state.structured_report.impressions:
                    reviews.append({'claim_id': claim_id(claim), 'visual_support': True,
                        'evidence_applicability': True, 'findings_impression_consistency': True,
                        'reporting_conventions': True, 'rationale': 'Synthetic integration assertion.',
                        'evidence_relations': [{'evidence_id': e.image_id, 'relation': 'fully_supported',
                            'rationale': 'Synthetic mock assessment.'} for e in effective_support(state, claim).image_evidence]})
                name, args = 'record_claim_review', {'reviews': reviews,
                    'reviewed_observation_ids': [state.observation_graph[0]['observation_id']]}
            if name:
                yield StreamEvent(EventType.TOOL_USE, {'tool_name': name, 'tool_input': args,
                                                     'tool_use_id': f'{stage}_{turn}'})
            else:
                yield StreamEvent(EventType.TEXT_DELTA, {'text': 'Completed synthetic stage.'})

    monkeypatch.setattr('src.core.orchestrator.build_agent', build)
    settings = {'data': {'output_dir': str(tmp_path)}, 'pipeline': {'max_iterations': 3}}
    events = [e async for e in run_pipeline(Image.new('RGB', (8, 8), 'white'),
                                           {'modality': 'CXR'}, settings, VLM())]
    errors = [e.data for e in events if e.type == EventType.TOOL_RESULT and e.data['is_error']]
    assert errors == []
    result = events[-1].data
    assert result['review_status'] == 'PASS'
    assert result['iterations'] == 1
    assert len(result['rendered_claim_spans']) == 2
    graph = holder['state'].dependency_graph
    finding = holder['state'].structured_report.findings[0]
    assert graph.nodes[finding.generating_operation_id]['payload']['tool_name'] == 'build_structured_finding'
    producers = [source for source, target in graph.edges if target == finding.finding_id
                 and graph.nodes[source]['kind'] == 'tool_call']
    assert producers == [finding.generating_operation_id]
