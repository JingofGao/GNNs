import torch
import argparse
import numpy as np
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import DataLoader


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.01,)
    parser.add_argument('--epochs', type=int, default=200,)
    parser.add_argument('--batch_size', type=int, default=1,)
    parser.add_argument('--weight_decay', type=float, default=1e-4,)
    parser.add_argument('--hidden', type=int, default=128,)
    parser.add_argument('--dataset', type=str, default="Pubmed", choices=["Cora", "Citeseer", "Pubmed"])
    parser.add_argument('--model', type=str, default="HopGNN", choices=["ChebNet", "GCN", "HopGNN"])
    args = parser.parse_args()

    # dataset
    dataset = Planetoid(root='../data', name=args.dataset)
    data_loader = DataLoader(dataset, batch_size=args.batch_size)

    # seed
    torch.manual_seed(0)

    # model
    if args.model == "ChebNet":
        from ChebNet.ChebNet import ChebNet
        model = ChebNet(in_channels=dataset[0].x.shape[-1],
                        hidden_channels=args.hidden,
                        out_channels=len(set(dataset[0].y)),
                        K=2, layers=2).cuda()
    elif args.model == "GCN":
        from GCN.GCN import GCN
        model = GCN(in_channels=dataset[0].x.shape[-1],
                    hidden_channels=args.hidden,
                    out_channels=len(set(dataset[0].y)),
                    layers=2).cuda()
    elif args.model == "HopGNN":
        from HopGNN.HopGNN import HopGNN
        model = HopGNN(in_channels=dataset[0].x.shape[-1],
                       hidden_channels=args.hidden,
                       out_channels=len(set(dataset[0].y)),
                       num_hop=6, inter_layer=2).cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    CE_Loss = torch.nn.CrossEntropyLoss()

    # training
    best_val_acc = 0.
    best_test_acc = 0.
    train_loss_set = []
    for epoch in range(args.epochs):
        model.train()
        train_loss = []
        for x,edge_index,y,train_mask,val_mask,test_mask,batch,ptr in data_loader:
            optimizer.zero_grad()

            X = x[1].cuda()
            Y = y[1].cuda()
            edge_index = edge_index[1].cuda()
            train_mask = train_mask[1].cuda()

            out = model(X, edge_index)
            loss = CE_Loss(out[train_mask], Y[train_mask])
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
        train_loss = np.mean(train_loss)
        train_loss_set.append(train_loss)

        model.eval()
        val_acc = []
        test_acc = []
        with torch.no_grad():
            for x, edge_index, y, train_mask, val_mask, test_mask, batch, ptr in data_loader:
                X = x[1].cuda()
                Y = y[1].cuda()
                edge_index = edge_index[1].cuda()
                val_mask = val_mask[1].cuda()
                test_mask = test_mask[1].cuda()

                out = model(X, edge_index)
                acc = torch.mean((torch.argmax(out[val_mask], dim=-1) == Y[val_mask]).float())
                val_acc += [acc.item()]*len(val_mask)
                acc = torch.mean((torch.argmax(out[test_mask], dim=-1) == Y[test_mask]).float())
                test_acc += [acc.item()]*len(test_mask)
            val_acc = np.mean(val_acc)
            test_acc = np.mean(test_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc

        print('Epoch [%d/%d], Loss: %.4f, Val Accuracy: %.2f%%, Test Accuracy: %.2f%%'
              % (epoch + 1, args.epochs, train_loss, 100 * val_acc, 100 * test_acc))
    print("%s (%s), acc: %.4f" % (args.model, args.dataset, best_test_acc))
