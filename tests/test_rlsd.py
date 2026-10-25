"""Execute pinned RLSD advantage, schedule and Teacher code on CPU tensors."""
import ast
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Dict, List, Optional
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from common import UPSTREAM, load
from run import config, assert_rlsd_only


def functions(relative, names, namespace):
    source = UPSTREAM/relative
    nodes = [n for n in ast.parse(source.read_text(encoding='utf-8')).body
             if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), namespace)
    return namespace


def losses(torch):
    scope = {'torch':torch, 'np':np,
        'verl_F':SimpleNamespace(masked_mean=lambda x,m:(x*m).sum()/m.sum())}
    functions('verl/trainer/ppo/core_algos.py', ['agg_loss','compute_policy_loss'], scope)
    return functions('verl/trainer/ppo/rlsd_utils.py', ['compute_rlsd_token_advantage'], scope)


@pytest.mark.parametrize('field,value', [('rlsd_lambda',0.), ('warmdown_steps',50),
    ('clip_eps',.5), ('skill_all',True)])
def test_original_webshop_rlsd_recipe_not_silently_changed(tmp_path, field, value):
    cfg = config('1.5b', tmp_path/'model', tmp_path/'run')
    cfg.algorithm.rlsd[field] = value
    with pytest.raises(RuntimeError, match='original WebShop RLSD'):
        assert_rlsd_only(cfg)


def test_grpo_term_cannot_be_disabled(tmp_path):
    from omegaconf import OmegaConf
    cfg = config('3b', tmp_path/'model', tmp_path/'run')
    OmegaConf.update(cfg, 'actor_rollout_ref.actor.pg_loss_coef', 0., force_add=True)
    with pytest.raises(RuntimeError, match='GRPO policy loss'):
        assert_rlsd_only(cfg)


def native_schedule():
    path = UPSTREAM/'verl/trainer/ppo/rlsd_ray_trainer.py'
    cls = next(n for n in ast.parse(path.read_text()).body
               if isinstance(n,ast.ClassDef) and n.name=='RLSDRayTrainer')
    method = next(n for n in cls.body if isinstance(n,ast.FunctionDef)
                  and n.name=='_get_rlsd_lambda')
    space = {}
    exec(compile(ast.Module(body=[method],type_ignores=[]),str(path),'exec'),space)
    return space['_get_rlsd_lambda']


@pytest.mark.parametrize('step,expected', [(0,.5),(1,.495),(50,.25),(99,.005),(100,0.),(150,0.)])
def test_actual_upstream_lambda_schedule(step, expected, tmp_path):
    cfg = config('1.5b', tmp_path/'model', tmp_path/'run')
    trainer = SimpleNamespace(rlsd_lambda_init=cfg.algorithm.rlsd.rlsd_lambda,
        rlsd_lambda_warmdown_steps=cfg.algorithm.rlsd.warmdown_steps)
    assert native_schedule()(trainer,step) == pytest.approx(expected)


@pytest.mark.parametrize('mix',[0.,.25,.5])
def test_native_sign_aware_clipped_token_advantage_mask_and_detach(mix):
    torch = pytest.importorskip('torch')
    native = losses(torch)['compute_rlsd_token_advantage']
    mask = torch.tensor([[0.,1.,1.,1.,0.],[1.,1.,1.,1.,0.],[1.,1.,1.,1.,0.]])
    seq = torch.tensor([[9.,2.,2.,2.,9.],[-2.,-2.,-2.,-2.,9.],[0.,0.,0.,0.,9.]],requires_grad=True)
    student = torch.full((3,5),-2.,requires_grad=True)
    teacher = torch.tensor([[-1.,-1.,-3.,-2.,-1.]]*3,requires_grad=True)
    result = native(seq,student,teacher,mask,rlsd_lambda=mix,rlsd_clip_eps=.2)
    a = torch.tensor([[2.],[-2.],[0.]])
    expected_weight = torch.exp(a.sign()*(teacher.detach()-student.detach())).clamp(.8,1.2)
    expected = a*((1-mix)+mix*expected_weight)*mask
    assert torch.allclose(result,expected) and not result.requires_grad
    assert torch.count_nonzero(result[:,4]) == 0 and torch.count_nonzero(result[2]) == 0
    if mix == 0.:
        assert torch.equal(result, a*mask)
    else:
        assert result[0,1] > result[0,2]  # Positive advantage, larger Teacher gap increases weight.
        assert result[1,1] > result[1,2]  # Negative advantage uses the opposite signed weight.
    assert seq.grad is teacher.grad is student.grad is None


def test_actual_native_fit_replaces_advantage_then_original_ppo_backpropagates():
    torch = pytest.importorskip('torch')
    native = losses(torch)
    path = UPSTREAM/'verl/trainer/ppo/rlsd_ray_trainer.py'
    cls = next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef)
               and n.name=='RLSDRayTrainer')
    fit = next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='fit')
    assignments = sorted([n for n in ast.walk(fit) if isinstance(n,ast.Assign)], key=lambda n:n.lineno)
    first = next(n.lineno for n in assignments if ast.unparse(n.targets[0])=='seq_advantages')
    last = next(n.lineno for n in assignments if ast.unparse(n.targets[0])=="batch.batch['advantages']"
                and ast.unparse(n.value)=='token_advantages')
    statements = [n for n in assignments if first <= n.lineno <= last]
    student = torch.nn.Parameter(torch.full((2,3),-1.))
    old = student.detach().clone()
    teacher = torch.tensor([[-.8,-1.2,-.7],[-1.1,-.9,-1.3]],requires_grad=True)
    mask = torch.tensor([[1.,1.,0.],[1.,1.,0.]])
    advantages = torch.tensor([[1.,1.,0.],[-1.,-1.,0.]])
    trainer = SimpleNamespace(global_steps=50, rlsd_clip_eps=.2,
        rlsd_lambda_init=.5, rlsd_lambda_warmdown_steps=100)
    trainer._get_rlsd_lambda = lambda step:native_schedule()(trainer,step)
    batch = SimpleNamespace(batch={'advantages':advantages,'old_log_probs':old,
                                  'teacher_log_probs':teacher,'response_mask':mask})
    scope = {'self':trainer,'batch':batch,'compute_rlsd_token_advantage':native['compute_rlsd_token_advantage']}
    exec(compile(ast.Module(body=statements,type_ignores=[]),str(path),'exec'),scope)
    actual = batch.batch['advantages']
    assert not actual.requires_grad and not torch.equal(actual,advantages)
    expected = native['compute_rlsd_token_advantage'](advantages,old,teacher,mask,.25,.2)
    assert torch.allclose(actual,expected)
    pg = native['compute_policy_loss'](old,student,actual,mask,cliprange=.2)[0]
    pg.backward()
    assert torch.allclose(student.grad,-actual*mask/mask.sum())
    assert teacher.grad is None and torch.count_nonzero(student.grad[:,2]) == 0
    before = student.detach().clone()
    torch.optim.SGD([student],lr=.1).step()
    assert student[0,0] > before[0,0] and student[1,0] < before[1,0]


def provider():
    scope = {'os':os, 'json':json, 'Dict':Dict, 'List':List, 'Optional':Optional}
    functions('verl/trainer/ppo/rlsd_utils.py',
        ['load_skill_mapping','load_skill_content','SkillProvider'],scope)
    return scope['SkillProvider'](str(UPSTREAM/'skills/webshop'),skill_all=False)


def test_official_static_skill_loading_and_keyword_selection():
    skills = provider()
    assert len(skills.skill_contents) == 8
    general = skills.skill_contents['general_skills']
    selected = skills.get_privileged_info_from_data_source('text','Need a shoe and a USB cable')
    assert selected.startswith(general)
    assert skills.skill_contents['footwear'] in selected
    assert skills.skill_contents['electronics'] in selected
    assert skills.get_privileged_info_from_data_source('text','unmatched objective') == general
    lock = load('upstream.json')
    assert all('skills/webshop/'+name in lock['unchanged_files']
        for name in skills.skill_mapping['skill_files'].values())
    assert 'skills/webshop/skill_mapping.json' in lock['unchanged_files']


class ByteTokenizer:
    pad_token_id = 0
    def encode(self, text, add_special_tokens=False):
        return [x+1 for x in text.encode('utf-8')]
    def decode(self, ids, skip_special_tokens=False):
        return bytes(int(x)-1 for x in ids if int(x)>0).decode('utf-8')


def teacher_functions(torch):
    class Data:
        @staticmethod
        def from_dict(tensors): return SimpleNamespace(batch=tensors)
    scope = {'torch':torch,'DataProto':Data,'SkillProvider':object}
    functions('verl/utils/model.py',['compute_position_id_with_mask'],scope)
    functions('verl/trainer/ppo/rlsd_ray_trainer.py',['build_teacher_batch'],scope)
    path = UPSTREAM/'verl/trainer/ppo/rlsd_ray_trainer.py'
    cls = next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='RLSDRayTrainer')
    method = next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_compute_teacher_log_probs')
    exec(compile(ast.Module(body=[method],type_ignores=[]),str(path),'exec'),scope)
    return scope


@pytest.mark.parametrize('budget',[32,8192])
def test_native_teacher_only_prefix_current_actor_and_response_alignment(budget):
    torch = pytest.importorskip('torch')
    from omegaconf import OmegaConf
    codec = ByteTokenizer()
    prompt = 'User: please buy a shoe'
    p = codec.encode(prompt)
    response = torch.tensor([[900,901,0]])
    ids = torch.tensor([[0,0]+p+response[0].tolist()])
    mask = torch.tensor([[0,0]+[1]*len(p)+[1,1,0]])
    batch = SimpleNamespace(batch={'input_ids':ids,'attention_mask':mask,'responses':response},
        non_tensor_batch={'gamefile':np.array([None],dtype=object),'data_source':np.array(['text'])})
    original = {key:value.clone() for key,value in batch.batch.items()}
    skill_provider = provider()
    scope = teacher_functions(torch)
    calls = []
    actor = SimpleNamespace(version=1.)
    def score(teacher_batch):
        calls.append(teacher_batch)
        return SimpleNamespace(batch={'old_log_probs':torch.full_like(response, actor.version, dtype=torch.float)})
    actor.compute_log_prob = score
    trainer = SimpleNamespace(actor_rollout_wg=actor, skill_provider=skill_provider, tokenizer=codec,
        config=SimpleNamespace(data=OmegaConf.create({'max_prompt_length':budget,'truncation':'error'})))
    first = scope['_compute_teacher_log_probs'](trainer,batch)
    actor.version = 2.
    second = scope['_compute_teacher_log_probs'](trainer,batch)
    assert torch.all(first==1.) and torch.all(second==2.)  # Not a fixed reference checkpoint.
    expected_text = '[Privileged Skill Information]\n'+skill_provider.get_privileged_info_from_prompt(prompt)+'\n\n'+prompt
    expected = codec.encode(expected_text)[-budget:]
    t = calls[0].batch
    actual = t['input_ids'][0,:budget][t['attention_mask'][0,:budget].bool()].tolist()
    assert actual == expected
    assert t['responses'] is response
    assert torch.equal(t['input_ids'][:,-3:],response)
    assert torch.equal(t['attention_mask'][:,-3:],mask[:,-3:])
    assert all(torch.equal(batch.batch[key],value) for key,value in original.items())


def test_validation_is_native_unprivileged_rollout_not_teacher_scoring():
    path = UPSTREAM/'verl/trainer/ppo/ray_trainer.py'
    cls = next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='RayPPOTrainer')
    method = ast.unparse(next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_validate'))
    assert 'multi_turn_loop' in method and 'is_train=False' in method
    assert 'teacher_log_probs' not in method and 'skill_provider' not in method
