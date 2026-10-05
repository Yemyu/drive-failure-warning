"""Independent numeric and artifact checks. No production metric imports."""
import math
import hashlib
import json
import datetime

def ranking(rows):
    """Aggregate exact tie groups before ranking; never call production AP."""
    groups = {}
    unknown = 0
    for score, label, _ in rows:
        if not math.isfinite(score) or label not in (None,0,1):
            raise ValueError('invalid ranking row')
        if label is None:
            unknown += 1
        else:
            g=groups.setdefault(score,[0,0]);g[0]+=1;g[1]+=label
    positives=sum(g[1] for g in groups.values())
    known=sum(g[0] for g in groups.values())
    seen=hits=0;ap=0.;curve=[]
    if positives:
        for threshold in sorted(groups,reverse=True):
            count,pos=groups[threshold];seen+=count;hits+=pos
            precision=hits/seen;recall=hits/positives
            ap+=pos*precision/positives
            if pos or seen == count:
                curve.append((recall,precision,threshold))
    points=[]
    for i in range(101):
        point=next((p for p in curve if p[0]>=i/100),None)
        points.append(dict(target_recall=i/100,actual_recall=point[0] if point else None,
                           precision=point[1] if point else None,threshold=point[2] if point else None))
    return dict(ap=ap if positives else None,known=known,positive=positives,unknown=unknown,points=points)

def linear_score(parameters, features):
    ordered=sorted(parameters['features'],key=lambda f:f['position'])
    if len(ordered)!=16 or [f['position'] for f in ordered]!=list(range(16)):
        raise ValueError('invalid fixed feature positions')
    score=float(parameters['intercept'])
    for f in ordered:
        value=features[f['source_name']]
        if value is None:value=f['imputation_mean']
        scale=float(f['standardization_scale'])
        if scale<=0:raise ValueError('invalid scale')
        values=[float(value),float(f['standardization_mean']),scale,float(f['coefficient'])]
        if not all(math.isfinite(x) for x in values):raise ValueError('non-finite linear input')
        score+=(values[0]-values[1])/scale*values[3]
    if not math.isfinite(score):raise ValueError('non-finite reconstructed score')
    return score

def compare_csv(rows, expected, label):
    if len(rows)!=len(expected):raise ValueError(f'{label}: row count differs')
    for i,(actual,wanted) in enumerate(zip(rows,expected)):
        if set(actual)!=set(wanted):raise ValueError(f'{label}: fields differ')
        converted={key:csv_value(actual[key],value) for key,value in wanted.items()}
        require_equal(converted,wanted,f'{label}/{i}')

def alert_burden(records, eligible_device_days, opportunity_events, event_start, event_end):
    if eligible_device_days<=0 or opportunity_events<0:raise ValueError('invalid denominator')
    devices={};keys=set();hits=set();outside=set();missing=0;tp=fp=u=0
    for row in records:
        serial=str(row['serial_number']);devices[serial]=devices.get(serial,0)+1
        label=row['label']
        if label not in (None,0,1):raise ValueError('invalid alert label')
        tp+=label==1;fp+=label==0;u+=label is None
        failure=row.get('first_failure_date')
        key=row.get('event_key')
        if not key and failure and event_start<=failure<=event_end:key=f'{serial}|{failure}'
        if key:keys.add(str(key))
        if row.get('event_hit'):
            if key:hits.add(str(key))
            else:missing+=1
        elif label==1 and key:outside.add(str(key))
    a=len(records);d=len(devices);h=len(hits)
    return dict(alerts=a,known_hit_alerts=tp,known_no_hit_alerts=fp,unknown_alerts=u,
        unknown_ratio=u/a if a else None,event_hits=h,event_keys_observed=len(keys),event_keys_missing=missing,
        opportunity_events=opportunity_events,event_recall=h/opportunity_events if opportunity_events else None,
        alerted_devices=d,first_alerts=d,later_alerts=a-d,repeat_alert_devices=sum(v>1 for v in devices.values()),
        later_alert_share=(a-d)/a if a else None,capture_status='captured' if h else 'no_captured_event',
        alerts_per_event=a/h if h else None,alerted_devices_per_event=d/h if h else None,
        eligible_device_days=eligible_device_days,alerts_per_1000_device_days=a*1000/eligible_device_days,
        known_precision=tp/(tp+fp) if tp+fp else None,unknown_precision_lower=tp/a if a else None,
        unknown_precision_upper=(tp+u)/a if a else None,
        outside_main_event_alerts=sum(r['label']==1 and not r.get('event_hit',0) for r in records),
        outside_main_event_keys=len(outside))

def prepare_outcome_dates(daily):
    dates={}
    for row in daily:
        day=datetime.date.fromisoformat(row['date'])
        if day>datetime.date(2023,12,31):continue
        if day in dates:raise ValueError('duplicate observed date')
        value=int(row['failure'])
        if value not in (0,1):raise ValueError('invalid failure')
        dates[day]=value
    first=min((day for day,value in dates.items() if value),default=None)
    return dates,first


def outcome_from_prepared(prepared, decision):
    dates,first=prepared
    t=datetime.date.fromisoformat(decision)
    if t not in dates or (first is not None and first<=t):
        raise ValueError('sealed score is not pre-failure observed day')
    horizon=[t+datetime.timedelta(days=i) for i in range(1,8)]
    if first in horizon:return 1
    if all(day in dates for day in horizon):return 0
    return None


def outcome_from_dates(daily, decision):
    return outcome_from_prepared(prepare_outcome_dates(daily),decision)

OUTPUTS = frozenset(('input_manifest.json', 'metrics.json', 'CHECKS.json',
                     'pr_curve.csv', 'calibration_bins.csv', 'burden.csv',
                     'pr_curves.svg', 'calibration_current_lr.svg'))
EDGES = (0., .0001, .0003, .001, .003, .01, .03, .1, .3, 1.)

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def require_equal(actual, expected, location='value'):
    if isinstance(expected, list):
        if not isinstance(actual,list) or len(actual)!=len(expected):
            raise ValueError(f'{location}: list lengths differ')
        for i,(left,right) in enumerate(zip(actual,expected)):
            require_equal(left,right,f'{location}/{i}')
    elif isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError(f'{location}: fields differ')
        for key in expected:
            require_equal(actual[key], expected[key], f'{location}/{key}')
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if isinstance(actual, bool) or not isinstance(actual, (int, float)) or not math.isfinite(actual):
            raise ValueError(f'{location}: non-finite or nonnumeric value')
        if isinstance(expected, int):
            equal = actual == expected
        else:
            equal = abs(actual-expected) <= 1e-10
        if not equal:
            raise ValueError(f'{location}: {actual!r} != {expected!r}')
    elif actual != expected or type(actual) is not type(expected):
        raise ValueError(f'{location}: value/type differs')

def check_artifacts(out, summary, facts, expected_inputs, root):
    if set(summary.get('outputs', {})) != OUTPUTS:
        raise ValueError('required output set differs')
    if {p.name for p in out.iterdir()} != OUTPUTS | {'summary.json'}:
        raise ValueError('directory output set differs')
    for name in OUTPUTS:
        path = out / name
        if path.is_symlink() or not path.is_file() or digest(path) != summary['outputs'][name]:
            raise ValueError(f'output hash mismatch: {name}')
    for alias, name in [('metrics_sha256','metrics.json'), ('checks_sha256','CHECKS.json'),
                        ('input_manifest_sha256','input_manifest.json')]:
        if summary.get(alias) != summary['outputs'][name]:
            raise ValueError(f'alias mismatch: {alias}')
    if facts.get('input_sha256') != expected_inputs:
        raise ValueError('source bindings differ from fresh verified preflight')
    if not expected_inputs:
        raise ValueError('source bindings are empty')
    for name, expected in expected_inputs.items():
        if digest(root / name) != expected:
            raise ValueError(f'source changed: {name}')

def calibration(rows):
    """Independent per-bin accumulation with stable softplus losses."""
    bins = [dict(n=0,k=0,y=0,u=0,p=0.,pk=0.,b=0.,loss=0.,devices=set(),positive=set()) for _ in range(9)]
    total_b = total_l = total_p = 0.
    positive_devices = set()
    for z, y, serial in rows:
        if not math.isfinite(z) or y not in (None, 0, 1):
            raise ValueError('invalid observation')
        e = math.exp(-abs(z))
        p = 1/(1+e) if z >= 0 else e/(1+e)
        i = sum(p >= edge for edge in EDGES[1:-1])
        b = bins[i]; b['n'] += 1; b['p'] += p; b['devices'].add(serial)
        if y is None:
            b['u'] += 1
            continue
        loss = (max(-z, 0) if y else max(z, 0)) + math.log1p(e)
        sq = (p-y)**2
        b['k'] += 1; b['y'] += y; b['pk'] += p; b['b'] += sq; b['loss'] += loss
        total_b += sq; total_l += loss; total_p += p
        if y:
            b['positive'].add(serial); positive_devices.add(serial)
    output = []
    for i,b in enumerate(bins):
        n,k,y,u = (b[x] for x in ('n','k','y','u'))
        output.append(dict(bin=i,lower=EDGES[i],upper=EDGES[i+1],include_upper=i==8,
            all_rows=n,known_rows=k,positive_rows=y,unknown_rows=u,
            unknown_ratio=u/n if n else None,
            distinct_devices=len(b['devices']),positive_devices=len(b['positive']),
            mean_probability_all=b['p']/n if n else None,mean_probability_known=b['pk']/k if k else None,
            sum_brier_known=b['b'],sum_logloss_known=b['loss'],known_rate=y/k if k else None,
            all_rate_lower=y/n if n else None,all_rate_upper=(y+u)/n if n else None,
            sparse_known_rows=0<k<20,sparse_positive_devices=n>0 and len(b['positive'])<20,empty_bin=n==0))
    k=sum(b['k'] for b in bins);y=sum(b['y'] for b in bins);u=sum(b['u'] for b in bins)
    prevalence=y/k if k else None
    constant_loss=(-(prevalence*math.log(prevalence)+(1-prevalence)*math.log1p(-prevalence)) if 0<prevalence<1 else 0.) if k else None
    return dict(bins=output,known_rows=k,positive_rows=y,positive_devices=len(positive_devices),unknown_rows=u,
        unknown_ratio=u/(k+u) if k+u else None,predicted_positive_known=total_p,observed_positive_known=y,
        predicted_observed_ratio_known=total_p/y if y else None,
        mean_probability_known=total_p/k if k else None,known_prevalence=prevalence,
        brier=total_b/k if k else None,log_loss=total_l/k if k else None,
        constant_zero_brier=prevalence,constant_prevalence_brier=prevalence*(1-prevalence) if k else None,
        constant_prevalence_log_loss=constant_loss)

def csv_value(value, expected):
    if expected is None:
        if value != '': raise ValueError('CSV null differs')
        return None
    if isinstance(expected,bool):
        if value not in ('True','False'): raise ValueError('CSV boolean differs')
        return value=='True'
    if isinstance(expected,int): return int(value)
    if isinstance(expected,float): return float(value)
    return value
