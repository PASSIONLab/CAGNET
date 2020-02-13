import torch
import torch.multiprocessing as mp

import gcn_distr
import gcn

def main():
    print("Executing gcn_distr.py...")
    outputs_gcn_distr = gcn_distr.main(4, True)
    print("Executing gcn.py...")
    outputs_gcn = gcn.main()

    print(outputs_gcn)
    print(outputs_gcn_distr)
    print(torch.allclose(outputs_gcn_distr, outputs_gcn, atol=1e-03, rtol=1e-03))

if __name__ == '__main__':
    main()
