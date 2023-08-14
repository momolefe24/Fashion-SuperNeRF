import logging
import sys
import argparse
def get_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument("--person", default="Inference Pipeline")
    opt = parser.parse_args()
    return opt

logging.basicConfig(
format="%(asctime)s %(levelname)s %(message)s",
level=logging.DEBUG,
stream=sys.stdout,
)
opt = get_opt()
print(opt.person)
logging.warning("Hello")
with open(f'transforms_{opt.person}.json', 'w') as f:
    f.write('${person}')