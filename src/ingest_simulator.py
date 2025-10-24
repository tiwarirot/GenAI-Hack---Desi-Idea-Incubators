import os, time, json
from src.data_generator import gen_raw_grinding, gen_clinker, gen_quality, gen_altfuel, gen_cross

OUT = os.path.join(os.path.dirname(__file__), '..', 'data')
os.makedirs(OUT, exist_ok=True)

def write_all():
    rg = gen_raw_grinding()
    rg.to_csv(os.path.join(OUT, 'raw_grinding.csv'), index=False)
    cl = gen_clinker()
    cl.to_csv(os.path.join(OUT,'clinker.csv'), index=False)
    q = gen_quality()
    q.to_csv(os.path.join(OUT,'quality.csv'), index=False)
    af = gen_altfuel()
    af.to_csv(os.path.join(OUT,'altfuel.csv'), index=False)
    cross = gen_cross()
    cross.to_csv(os.path.join(OUT,'cross.csv'), index=False)
    print('[ingest] synthetic CSVs written to data/')

def stream_simulator(process='raw_grinding', batch_size=50, delay=0.05):
    mapping = {
        'raw_grinding': gen_raw_grinding,
        'clinker': gen_clinker,
        'quality': gen_quality,
        'altfuel': gen_altfuel,
        'cross': gen_cross
    }
    gen = mapping.get(process, gen_raw_grinding)
    df = gen(batch_size)
    for _, row in df.iterrows():
        yield row.to_dict()
        time.sleep(delay)
