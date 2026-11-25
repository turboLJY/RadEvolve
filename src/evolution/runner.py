"""Real reporting-pipeline adapter, including serializable pre-stage checkpoints."""
from __future__ import annotations

import base64
from copy import deepcopy
from dataclasses import fields, is_dataclass, asdict
from datetime import datetime
from enum import Enum
import hashlib
import io
from pathlib import Path

from .evaluation import RunOutput
from .runtime import RuntimeBundle
from .schemas import Checkpoint, Harness, SkillLibrary, Versions, content_hash


def encode_state(value):
    """JSON codec with an explicit allowlist, never pickle or dynamic imports."""
    from PIL import Image
    from src.core.provenance import DependencyGraph
    if isinstance(value, Enum):
        return {'__type__': type(value).__name__, 'value': value.value}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, datetime):
        return {'__type__': 'datetime', 'value': value.isoformat()}
    if isinstance(value, Image.Image):
        buffer = io.BytesIO()
        value.save(buffer, format='PNG')
        return {'__type__': 'image', 'value': base64.b64encode(buffer.getvalue()).decode()}
    if isinstance(value, DependencyGraph):
        return {'__type__': 'DependencyGraph', 'value': value.to_dict()}
    if isinstance(value, (set, frozenset, tuple)):
        return {'__type__': type(value).__name__, 'value': [encode_state(v) for v in value]}
    if isinstance(value, list):
        return [encode_state(v) for v in value]
    if isinstance(value, dict):
        return {str(k): encode_state(v) for k, v in value.items()}
    if is_dataclass(value) and type(value).__module__ == 'src.core.types':
        return {'__type__': type(value).__name__, 'fields': {f.name: encode_state(getattr(value, f.name)) for f in fields(value)}}
    raise TypeError(f'Unsupported checkpoint state type: {type(value).__name__}')


def decode_state(value):
    from PIL import Image
    from src.core import types
    from src.core.provenance import DependencyGraph
    if isinstance(value, list):
        return [decode_state(v) for v in value]
    if not isinstance(value, dict):
        return value
    tag = value.get('__type__')
    if not tag:
        return {k: decode_state(v) for k, v in value.items()}
    if tag == 'datetime':
        return datetime.fromisoformat(value['value'])
    if tag == 'image':
        return Image.open(io.BytesIO(base64.b64decode(value['value']))).copy()
    if tag == 'DependencyGraph':
        return DependencyGraph.from_dict(value['value'])
    if tag in {'set', 'frozenset', 'tuple'}:
        return {'set': set, 'frozenset': frozenset, 'tuple': tuple}[tag](decode_state(v) for v in value['value'])
    allowed = {name: cls for name, cls in vars(types).items()
               if isinstance(cls, type) and cls.__module__ == 'src.core.types'
               and (is_dataclass(cls) or issubclass(cls, Enum))}
    if tag not in allowed:
        raise ValueError(f'Prohibited checkpoint object type: {tag}')
    cls = allowed[tag]
    return cls(value['value']) if issubclass(cls, Enum) else cls(**{k: decode_state(v) for k, v in value['fields'].items()})


def encode_image_store(store) -> dict:
    return {'session_id': store.session_id, 'max_views': store.max_views,
            'images': encode_state(store._images), 'tool_chains': deepcopy(store._tool_chains),
            'metadata': deepcopy(store._metadata), 'view_keys': deepcopy(store._view_keys), 'counter': store._counter,
            'coarse_inspected': encode_state(getattr(store, '_coarse_inspected', set()))}


def decode_image_store(value):
    from src.data.image_store import SessionImageStore
    store = SessionImageStore(value['session_id'], value['max_views'])
    store._images = decode_state(value['images'])
    store._tool_chains = deepcopy(value['tool_chains'])
    store._metadata = deepcopy(value['metadata'])
    store._view_keys = deepcopy(value['view_keys'])
    store._counter = value['counter']
    store._coarse_inspected = decode_state(value.get('coarse_inspected', {'__type__': 'set', 'value': []}))
    return store


class CheckpointCaptured(Exception):
    def __init__(self, checkpoint: Checkpoint):
        self.checkpoint = checkpoint


class PipelineRunner:
    def __init__(self, settings: dict, llm_client, *, pipeline=None):
        from src.core.orchestrator import run_pipeline
        self.settings = deepcopy(settings)
        self.llm_client = llm_client
        self.pipeline = pipeline or run_pipeline
        self.settings.setdefault('retrieval', {})['require_snapshot'] = True
        corpus = self.settings['retrieval'].get('snapshot_path')
        if not corpus or not Path(corpus).is_file():
            raise ValueError('Offline evolution requires a materialized retrieval snapshot')
        from src.kb.snapshot import CorpusSnapshot
        self.snapshot_version = CorpusSnapshot(corpus, self.settings['retrieval'].get('snapshot_version', '')).version
        self.settings['retrieval']['snapshot_version'] = self.snapshot_version
        self._corpus_file_hash = hashlib.sha256(Path(corpus).read_bytes()).hexdigest()
        model = asdict(llm_client.config)
        model.pop('api_key', None)
        self.model_version = content_hash(model)
        self.resource_version = content_hash({key: self.settings.get(key, {}) for key in ('pipeline', 'agents', 'limits', 'accounting')})
        self.run_records: list[dict] = []

    def _verify(self, versions: Versions) -> None:
        if versions.model != self.model_version or versions.resource_limits != self.resource_version or versions.retrieval_snapshot != self.snapshot_version:
            raise ValueError('Runner configuration differs from the fixed selection versions')
        if hashlib.sha256(Path(self.settings['retrieval']['snapshot_path']).read_bytes()).hexdigest() != self._corpus_file_hash:
            raise ValueError('Retrieval corpus changed during evolution')
        model = asdict(self.llm_client.config)
        model.pop('api_key', None)
        if content_hash(model) != self.model_version:
            raise ValueError('Execution model changed during selection')

    def _input(self, case_input: dict):
        from PIL import Image
        from .manifest import DevelopmentCase
        # Reuse context validation even when an adapter is called independently.
        DevelopmentCase(case_input['case_id'], case_input['patient_id'], 'runtime', 'train', 'evolution',
                        case_input['image_path'], case_input['clinical_context'], ())
        image_path = Path(case_input['image_path'])
        context = deepcopy(case_input['clinical_context'])
        if context.get('sequence_paths') and not context.get('image_paths'):
            context['image_paths'] = list(context['sequence_paths'])
        if context.get('sequences') and not context.get('image_paths'):
            context['image_paths'] = [item['path'] for item in context['sequences']]
        if str(image_path).endswith(('.nii', '.nii.gz')):
            from src.data.chest_ct_jsonl_loader import load_nifti_preview
            modality = context.get('modality', '')
            window = 'mri' if modality in {'MRI', 'brain_mri'} else 'soft_tissue' if modality == 'abdominal_ct' else 'lung'
            image, metadata = load_nifti_preview(image_path, window=window)
            context['preview_metadata'] = metadata
            context['volume_path'] = str(image_path.resolve())
            context.setdefault('preview_window', window)
            context.setdefault('image_paths', [str(image_path.resolve())])
        else:
            image = Image.open(image_path).convert('RGB')
        paths = {str(image_path.resolve())}
        def collect(value, key=''):
            if isinstance(value, dict):
                for child_key, item in value.items():
                    collect(item, child_key)
            elif isinstance(value, list):
                for item in value:
                    collect(item, key)
            elif isinstance(value, str) and (key in {'path', 'file', 'image_paths', 'sequence_paths', 'prior_images'} or key.endswith('_path')):
                paths.add(str(Path(value).resolve()))
        collect(context)
        hashes = {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in sorted(paths)}
        digest = content_hash({'input': case_input, 'image_sources': hashes})
        return image, context, digest

    async def checkpoint(self, case_input: dict, stage: str, harness: Harness,
                         skills: SkillLibrary, versions: Versions) -> Checkpoint:
        self._verify(versions)
        image, context, digest = self._input(case_input)
        bundle = RuntimeBundle(deepcopy(harness), deepcopy(skills), versions.to_dict())
        def capture(current_stage, state, store):
            if current_stage != stage:
                return
            checkpoint = Checkpoint(case_input['case_id'], stage, versions,
                                    [encode_state(state.conversation_history)],
                                    {'visual_findings_text': state.visual_findings_text, 'evidence_summary': state.evidence_summary},
                                    deepcopy(state.observation_graph),
                                    [item.to_dict() for item in [*state.image_evidence, *state.retrieved_evidence]],
                                    state.dependency_graph.to_dict(), deepcopy(state.execution_trace),
                                    {'pipeline': encode_state(state), 'image_store': encode_image_store(store), 'input_hash': digest})
            raise CheckpointCaptured(checkpoint)
        try:
            async for _ in self.pipeline(image, context, self.settings, self.llm_client,
                                         runtime_bundle=bundle, checkpoint_callback=capture):
                pass
        except CheckpointCaptured as captured:
            self.run_records.append({'kind': 'checkpoint_construction', 'case_id': case_input['case_id'],
                                     'versions': versions.to_dict(), 'request_records': decode_state(captured.checkpoint.state['pipeline']).request_records})
            return captured.checkpoint
        raise RuntimeError('Pipeline completed without reaching the selected pre-decision stage')

    async def run(self, case_input: dict, harness: Harness, skills: SkillLibrary,
                  versions: Versions, checkpoint: Checkpoint | None = None) -> RunOutput:
        from src.core.types import EventType
        self._verify(versions)
        if versions.harness != harness.version or versions.skills != skills.version:
            raise ValueError('Run versions do not identify supplied harness/library')
        image, context, digest = self._input(case_input)
        kwargs = {'runtime_bundle': RuntimeBundle(deepcopy(harness), deepcopy(skills), versions.to_dict())}
        if checkpoint:
            if checkpoint.case_id != case_input['case_id'] or checkpoint.state['input_hash'] != digest:
                raise ValueError('Checkpoint belongs to different case inputs')
            fixed = ('model', 'retrieval_snapshot', 'evaluator', 'resource_limits', 'skills')
            if any(getattr(checkpoint.versions, key) != getattr(versions, key) for key in fixed):
                raise ValueError('Checkpoint fixed versions are stale')
            if checkpoint.versions.harness not in {harness.version, harness.parent_version}:
                raise ValueError('Checkpoint does not precede this control/candidate intervention')
            state = decode_state(checkpoint.state['pipeline'])
            store = decode_image_store(checkpoint.state['image_store'])
            kwargs.update(initial_state=state, image_store=store, start_stage=checkpoint.before_stage)
        result = None
        error_events = []
        executed_requests = []
        upstream_requests = list(state.request_records) if checkpoint else []
        try:
            async for event in self.pipeline(image, context, self.settings, self.llm_client, **kwargs):
                if event.type == EventType.STATUS and event.data.get('request_record'):
                    executed_requests.append(deepcopy(event.data['request_record']))
                if event.type == EventType.ERROR:
                    error_events.append(event.data)
                if event.type == EventType.DONE and 'structured_report' in event.data:
                    result = deepcopy(event.data)
        except Exception as exc:
            error_events.append({'error_type': type(exc).__name__, 'message': str(exc)})
        result = result or {}
        request_records = result.get('request_records') or [*upstream_requests, *executed_requests]
        costs = [row.get('cost_cny') for row in request_records]
        cost = sum(costs) if costs and all(value is not None for value in costs) else None
        complete = bool(result.get('completed')) and bool(result.get('structured_report')) and not error_events
        within = (result.get('iterations', 0) <= self.settings.get('pipeline', {}).get('max_iterations', 3)
                  and len(result.get('images', {})) <= self.settings.get('limits', {}).get('max_image_views', 32))
        run = RunOutput(result, complete, cost, within, {'errors': error_events, 'request_records': request_records,
                                                       'versions': versions.to_dict(), 'from_checkpoint': checkpoint is not None,
                                                       'upstream_request_count': len(upstream_requests),
                                                       'executed_requests': executed_requests,
                                                       'executed_cost_cny': sum(r['cost_cny'] for r in executed_requests) if executed_requests and all(r.get('cost_cny') is not None for r in executed_requests) else None})
        self.run_records.append({'kind': 'replay' if checkpoint else 'complete_run', 'case_id': case_input['case_id'],
                                 'complete': complete, 'cost_cny': cost, 'trace': run.trace, 'report': result})
        return run
