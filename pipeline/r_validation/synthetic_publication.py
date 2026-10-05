"""Exclusive completion markers; source access authorization is separate."""
import os
import signal
import math

from pipeline.reproducible.artifacts import write_manifest


def publish_synthetic(output, *, artifacts, guard, cancel, validate, calendar):
    return _publish(output, artifacts=artifacts, guard=guard, cancel=cancel, validate=validate,
                    name='synthetic_complete', metadata={'schema':'r-synthetic-completion-v1',
                    'status':'synthetic_complete','profile':'self_generated_zip_fixture','calendar':calendar,
                    'real_data_release':False,'model_adoption':False})


def publish_bound(output, *, artifacts, guard, cancel, validate, contract, evaluation):
    from pipeline.r_validation.bound_zip import validate_bound_zip
    validate_bound_zip(contract)
    real = contract['profile'] in {'released_zip', 'released_q1_zip'}
    protocol = evaluation['protocol_metrics']
    reasons = [reason for reason in protocol['interval_publication']['blocking_reasons']
               if reason != 'complete_acceptance_pending']
    if not real: reasons.append('synthetic_evidence_only')
    interval = protocol['bootstrap']['interval'] if not reasons else None
    if not reasons and (not isinstance(interval,list) or len(interval)!=2
                       or any(type(v) not in (float,int) or not math.isfinite(v) for v in interval)
                       or interval[0]>interval[1]):
        raise ValueError('publishable interval is missing or invalid')
    numerical=protocol['interpretation_gate']['numerical_conditions_met']
    if type(numerical) is not bool: raise ValueError('gain decision must be boolean')
    return _publish(output, artifacts=artifacts, guard=guard, cancel=cancel, validate=validate,
                    name='r_complete' if real else 'bound_synthetic_complete', metadata={
                    'schema':'r-bound-completion-v1','status':'complete' if real else 'synthetic_complete',
                    'profile':contract['profile'],'calendar':'quarter',
                    'real_data_release':real,'model_adoption':False,'source_contract':contract,
                    'interval_publication':{'status':'published' if not reasons else 'withheld',
                                            'interval_pp':interval,'blocking_reasons':reasons},
                    'gain_evidence':{'complete_acceptance':'passed',
                        'numerical_conditions_met':numerical,
                        'support_gain':real and not reasons and numerical,
                        'final_interpretation':'pending_review'}})


def _publish(output, *, artifacts, guard, cancel, validate, name, metadata):
    staged=output/(name+'.staged.json')
    final=output/(name+'.json')
    manifest=write_manifest(staged,{**metadata,
        'verification':metadata.get('verification','independent_recomputation_passed'),
        'resource_sample_until':'commit_decision','commit_point':'exclusive_hard_link'},artifacts=artifacts)
    validate()
    guard.check_or_raise()
    guard.stop()
    if guard.violation is not None: raise RuntimeError('resource guard before synthetic publication: '+guard.violation)
    if not hasattr(signal,'pthread_sigmask'): raise RuntimeError('atomic signal decision unsupported')
    previous=signal.pthread_sigmask(signal.SIG_BLOCK,{signal.SIGINT,signal.SIGTERM})
    try:
        if cancel.is_set() or signal.sigpending() & {signal.SIGINT,signal.SIGTERM}:
            raise RuntimeError('cancelled before synthetic publication')
        # A signal arriving after this decision belongs to the completed commit.
        # link is atomic and refuses to replace a competing target.
        os.link(staged,final)
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK,previous)
    try:
        staged.unlink()
    except OSError:
        pass  # The completed commit is not retracted by staged-file cleanup.
    return manifest
