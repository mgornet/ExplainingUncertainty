import matplotlib # Importing matplotlib for it working on remote server
import matplotlib.pyplot as plt
import matplotlib.colors as color

import numpy as np

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# BUILD
##################################################################################

class BasicBlock3(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock3, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.activ = nn.ReLU()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = self.activ(self.conv1(x[0]))
        out = self.conv2(out)
        out += self.shortcut(x[0])
        out = self.activ(out)
        return [out, x[0]]

class ResNet_34(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, in_maps=1):
        super(ResNet_34, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(in_maps, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.fc = nn.Linear(512*block.expansion, num_classes)
        self.activ = nn.ReLU()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        #out = F.relu(self.bn1(self.conv1(x))) # For ResNet with BatchNorm layer
        out = self.activ(self.conv1(x))
        out = self.layer1([out, out])
        out = self.layer2(out)
        out_inter1 = self.layer3(out)
        out = self.layer4(out_inter1)
        out_inter1, out_inter2 = out_inter1[0], out[1]
        out = out[0]
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return self.activ(out), [out_inter1, out_inter2]

class MyResNet(nn.Module):
    def __init__(self, version='4', bn=False, n_classes=10, in_maps=1):
        super(MyResNet, self).__init__()
        self.convnet = ResNet_34(BasicBlock3, [2, 2, 2, 2], num_classes=n_classes, in_maps=in_maps)

        num_ftrs = self.convnet.fc.in_features
        self.fc2 = nn.Linear(num_ftrs, n_classes)
        self.convnet.fc = nn.Linear(num_ftrs, num_ftrs) # 200 is the number of output classes in CUB_2011/200 dataset

    def forward(self, inp):
        output, output_intermediate = self.convnet(inp)     
        return self.fc2(output), output_intermediate
    
    
class attr_RN18_multi(nn.Module):
    def __init__(self, out_size=49*2, in_maps1=256, in_maps2=512):
        super(attr_RN18_multi, self).__init__()
        self.conv1 = nn.Conv2d(in_maps1, 64, kernel_size=3, padding=1, stride=2)
        self.conv2 = nn.Conv2d(in_maps2, 64, kernel_size=3, padding=1, stride=1)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1, stride=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64*3*3, out_size, bias=True)
        #self.activ = nn.LeakyReLU()
        self.activ = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d(3)

    def forward(self, inter_list):
        x1 = self.activ( self.conv1(inter_list[0]) )
        x2 = self.activ( self.conv2(inter_list[1]) )
        x = torch.cat((x1, x2), 1)
        x = self.activ( self.conv3(x) )
        x = self.pool(x).view(-1, 64*3*3)
        #x = self.fc1(x) # Till model 9
        #x = x.view(-1, 64*3*)
        x = self.activ(self.fc1(x))
        #x = self.fc1(x)
        return x

class decode_CIFAR(nn.Module):
    def __init__(self, in_size=64*2):
        super(decode_CIFAR, self).__init__()
        self.fc1 = nn.Linear(in_size, 64*10, bias=True)
        self.trconv1 = nn.ConvTranspose2d(10, 24, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True)
        self.trconv2 = nn.ConvTranspose2d(24, 8, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True)
        self.fc2 = nn.Linear(8*32*32, 3*32*32)
        self.activ = nn.ReLU()

    def forward(self, inp):
        x = self.fc1(inp)
        x = x.view(-1, 10, 8, 8)
        x = self.activ( self.trconv1(x) )
        x = self.activ( self.trconv2(x) )
        x = x.view(-1, 8*32*32)
        x = self.fc2(x).view(-1, 3, 32, 32)
        return x
    
class explainer(nn.Module):
    def __init__(self, in_size=30, max_elem=15, n_classes=200):
        super(explainer, self).__init__()
        self.max_elem = max_elem
        self.fc1 = nn.Linear(in_size, n_classes, bias=True)
        self.drop = nn.Dropout(0.01)

    def forward(self, inp):
        # Select the max elems by multiplying input by the appropriate tensor
        x = self.drop(inp)
        return self.fc1(x)

# ANALYZE
##################################################################################

def analyze(f, g, h, d, device, test_loader, location):
    f.eval(), g.eval(), h.eval(), d.eval()
    f, g, h, d = f.to(device), g.to(device), h.to(device), d.to(device)
    conf_matx_fy = np.zeros([10, 10]) # n_classes x n_classes
    conf_matx_hf = np.zeros([10, 10])
    conf_matx_hy = np.zeros([10, 10])
    for batch_info in test_loader:
        data, target = batch_info[0].to(device), batch_info[1].to(device)
        output, inter = f(data)
        embed = g(inter)
        rec_data, expl = d(embed), h(embed)
        pred_f = output.argmax(dim=1).cpu().data.numpy()
        pred_h = expl.argmax(dim=1).cpu().data.numpy()
        y = target.cpu().data.numpy()
        for j in range(y.shape[0]):
            conf_matx_fy[pred_f[j], y[j]] += 1
            conf_matx_hf[pred_h[j], pred_f[j]] += 1
            conf_matx_hy[pred_h[j], y[j]] += 1

    return conf_matx_fy, conf_matx_hf, conf_matx_hy

# INTERPRET
##################################################################################

def collect_g_data(f, g, h, device, data, subset=False):
    f.eval(), g.eval(), h.eval()
    f, g, h = f.to(device), g.to(device), h.to(device)
    weights = h.fc1.weight.cpu().data.numpy()
    g_data = []
    all_y = []
    num_batch = 0
    subset_data = []
    expl_data = [] # Only append data in this if subset is true, else it'll possibly increase the time by a lot
    expl_pred = []
    if not subset:
        dataloader = torch.utils.data.DataLoader(data, batch_size=16*4, shuffle=False, num_workers=64)
    else:
        dataloader = torch.utils.data.DataLoader(data, batch_size=20, shuffle=True, num_workers=20)
    for batch_info in dataloader:
        num_batch += 1
        data, target = batch_info[0].to(device), batch_info[1].to(device)                                 
        output, inter = f(data)
        embed = g(inter)
        expl = np.zeros(embed.shape)
        pred = h(embed).argmax(dim=1).cpu().data.numpy()
        g_data.append(embed.cpu().data.numpy())
        all_y += list(target.cpu().data.numpy())
        expl_pred += list(h(embed).argmax(dim=1).cpu().data.numpy())
        if subset:
            subset_data.append(data)
            for i in range(pred.shape[0]):
                expl[i] = embed[i].cpu().data.numpy() * weights[pred[i]]
                expl[i] = expl[i]/expl[i].max()
            expl_data.append(expl)
            if num_batch > 50:
                subset_data = torch.cat(subset_data).unsqueeze(dim=1) #unsqueeze is done to make code in save image functions compatible with shape of subset_data
                break
    g_data = np.concatenate(g_data)
    if subset:
        expl_data = np.concatenate(expl_data)
    return g_data, np.array(all_y), subset_data, expl_data, np.array(expl_pred)

def sparse_sense(f, g, h, data, device, mults):
    f, g, h = f.eval(), g.eval(), h.eval()
    gdata = collect_g_data(f, g, h, device, data)[0]
    pred = h(torch.tensor(gdata).to(device)).argmax(dim=1).cpu().data.numpy()
    weights  = h.fc1.weight.cpu().data.numpy()
    result = []
    for multiplier in mults:
        sparse = 0
        for i in range(pred.shape[0]):
            expl_vec = np.abs(gdata[i] * weights[pred[i]])
            thresh = np.abs(expl_vec).max() / multiplier
            sparse += np.sum(expl_vec > thresh)
        #print (sparse/pred.shape[0])
        result.append(sparse/pred.shape[0])
    return result

def grad_inp_embed(f_gb, g, device, inp, embed_idx, dataset='mnist'):
    # Computes appropriate saliency map for an attribute w.r.t input
    # Assume inp of shape 1 x 28 x 28
    g = g.eval()
    inp = inp.unsqueeze(0)
    g, inp = g.to(device), inp.to(device)
    inp.requires_grad = True
    if dataset == 'qdraw':
        output, inter = f_gb.model(inp)
    else:
        output, inter = f_gb(inp)
    if dataset == 'qdraw':
        f_gb.model.zero_grad()
    else:
        f_gb.zero_grad()
    embed = g(inter)
    if dataset == 'mnist' or dataset == 'fmnist' or dataset == 'qdraw':
        grad = torch.autograd.grad(embed[0, embed_idx], inp)[0][0, 0].cpu().data.numpy()
    elif dataset == 'cifar10':
        grad = torch.autograd.grad(embed[0, embed_idx], inp)[0][0].abs().sum(dim=0).cpu().data.numpy() # Add the code to shift axes, then remove this comment
    #print (grad.shape)
    return grad

def optimize_inp(f, g, embed_idx, device, inp_shape=[1, 1, 28, 28], init=None, max_val=1.0, min_val=0.0, lmbd_tv=1.0, lmbd_bound=1.0, C=1.0, lmbd_l1=0):
    # Function to run activation maximization with partial initialization
    # initialize input with input shape and make requires_grad True
    f, g = f.eval().to(device), g.eval().to(device) 
    inp = torch.empty(inp_shape).to(device)
    if init is None:
        #4.0 * (nn.init.uniform_(inp) - 0.5) # Initialization line
        nn.init.uniform_(inp)
    else:
        inp = 1.0 * init
    inp.requires_grad = True
    new_lr = 0.05
    inp.to(device)
    for epoch in range(6):
        #optimizer = optim.SGD([inp], lr=new_lr, momentum=0.9)
        optimizer = optim.Adam([inp], lr=new_lr)
        new_lr = new_lr/2
        for i in range(50):
            optimizer.zero_grad()
            output, inter = f(inp)
            embed = g(inter)

            loss_l1 = (inp.abs()).mean()
            loss_bound = (( (inp > max_val).float() + (inp < min_val).float() )*(inp.abs())).mean()
            loss_tv = (inp[:, :, 0:inp_shape[2]-1, :] - inp[:, :, 1:inp_shape[2], :]).abs().mean() + (inp[:, :, :, 0:inp_shape[3]-1] - inp[:, :, :, 1:inp_shape[3]]).abs().mean()
            loss = C*embed[:, embed_idx].sum() - lmbd_l1 * loss_l1 - lmbd_bound * loss_bound - lmbd_tv * loss_tv

            loss.backward()
            inp.grad = -1 * inp.grad
            optimizer.step()
            if (i % 51 == 0 and i == 3):
                print (epoch, loss.item(), embed[:, embed_idx].sum(), loss_l1.item(), loss_bound.item(), loss_tv.item())
    return inp.cpu().data.numpy()

def save_expl_images_class_cifar(indices, data, gdata, f, f_copy, g, device, dataset, model_name='', d=None):
    # This function assumed specific shape of indices
    if dataset == 'qdraw' or dataset == 'cifar10':
        f_gb = gb.GuidedBackprop(f)
    else:
        f_gb = f
    f_copy = f_copy.eval()
    for i in range(indices.shape[2]): # Fixing the attribute (coordinate of attribute vector)
        for j in range(indices.shape[0]): # Fixing the class
            for k in range(indices.shape[1]):
                if indices[j, k, i] == -1:
                    continue
                img = unnorm(data[indices[j, k, i]][0].cpu().data.numpy(), norm_mean, norm_std)
                init_img = 0.4*data[indices[j, k, i]] 
                cur_img = optimize_inp(f_copy, g, i, device, list(init_img.shape), max_val=2.5, min_val=-2.5, init=init_img, lmbd_bound=20.0, lmbd_tv=20.0, C=2.0, lmbd_l1=0.0)
                grad = grad_inp_embed(f_gb, g, device, data[indices[j, k, i]][0], i, dataset=dataset)
                #if gdata[indices[j, k, i], i] < gdata[:, i].max()/4.0:
                    #continue
                fig = plt.figure()
                fig.add_subplot(1, 2, 1)
                plt.imshow(img)
                plt.axis('off')
                #fig.add_subplot(1, 5, 2)
                #plt.imshow(grad)
                #plt.axis('off')
                #fig.add_subplot(1, 5, 3)
                #attr = g(f(data[indices[j, k, i]][0].unsqueeze(0))[1])
                ######attr[:, i] = 0
                #plt.imshow(unnorm(d(attr)[0].cpu().data.numpy(), norm_mean, norm_std))
                #plt.axis('off')
                #attr[:, i] = 0
                #fig.add_subplot(1, 5, 4)
                #plt.imshow(unnorm(d(attr)[0].cpu().data.numpy(), norm_mean, norm_std))
                #plt.axis('off')
                fig.add_subplot(1, 2, 2)
                plt.imshow(unnorm(cur_img[0], norm_mean, norm_std).mean(axis=2) )
                plt.axis('off')
                #fig.add_subplot(1, 6, 6)
                #plt.imshow(unnorm(cur_img[0], np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0])) )
                #plt.axis('off')
                fig.subplots_adjust(wspace=0.04) 
                plt.savefig('output/' + dataset + '_output/explanation_images_' + model_name  + '/attr' + str(i) + '_class' + str(j) + '_' + str(k), bbox_inches='tight', pad_inches = 0.03)
                plt.close()
                
    return

def extract_attr_class_max(gdata, all_y, expl_data, expl_pred, n_idx=1, thresh=0.1):
    n_class = all_y.max() + 1
    indices = np.zeros([n_class, n_idx, gdata.shape[1]])
    true_max = np.max(gdata, axis=0)
    for i in range(n_class):
        pos_arr = np.where(all_y == i)[0]
        gdata_class = gdata[pos_arr] # Data for the ith class
        local_idx = np.argsort(-gdata_class, axis=0)[:n_idx] 
        indices[i] = pos_arr[local_idx]
    if len(expl_data > 0):
        indices2 = np.zeros([n_class, n_idx, gdata.shape[1]]) - 1
        expl_class = np.zeros([n_class, gdata.shape[1]])
        for i in range(n_class):
            pos_arr = np.where(expl_pred == i)[0]
            expl_class[i] = expl_data[pos_arr].mean(axis=0)
            select_attr = np.where(expl_class[i] > thresh)[0]
            for attr in select_attr:
                indices2[i, :, attr] = indices[i, :, attr]
        return indices2.astype(int), expl_class 
    return indices.astype(int)

def generate_model_explanations(f, g, h, d, data, device, dataset, checkpoint, model_name='', subset=False):
    if not subset:
        print ('Collecting attribute vectors on the given data')
    else:
        print ('Collecting attribute vectors on random subset of the given data')
    gdata, all_y, subset_data, expl_data, expl_pred = collect_g_data(f, g, h, device, data, subset=subset)
    indices2, rel = extract_attr_class_max(gdata, all_y, expl_data, expl_pred, 3, thresh=0.1)
    try:
        os.mkdir('output/' + dataset + '_output/explanation_images_' + str(model_name))	
        #os.mkdir('output/' + dataset + '_output/explanation_images_' + str(model_name) + '/inp_optimize')
    except:
        print ('Writing images in an old folder. May overwrite some files')
    print ('Saving images')
    if not subset:
        subset_data = data
    f_copy = MyResNet(version='34', bn=False, n_classes=10, in_maps=3).to(device)
    f_copy.load_state_dict(checkpoint['f_state_dict'])
    
    #save_expl_images_class(indices2, subset_data, gdata, f, f_copy, g, device, dataset, str(model_name), d)
    return rel

def plot_rel(rel, dataset, classes):
    rel2 = 1.0*rel
    class_name = classes
    x_pos = np.array(range(rel.shape[1])).astype(str)
    x_pos = np.array(['$\phi_{'+i+'}$' for i in x_pos])
    x_pos_int = np.array([i for i in np.array(range(rel.shape[1]))])

    plt.figure(figsize=(10, 10))
    plt.imshow(rel.T, cmap='hot', aspect=0.6)
    plt.xticks(list(range(rel.shape[0])), class_name, fontsize=15, rotation=45)   
    plt.yticks(list(range(rel.shape[1])), range(rel.shape[1]), fontsize=12, rotation=0)
    plt.ylabel('Attribute number', fontsize=20)
    plt.title('Class-attribute relevances', fontsize=21)
    plt.colorbar()
    plt.show()
    
    return

def analyze_img(f, g, h, d, test_data, classes, device, idx, n_attr=3, save=False, location=None):
    f_copy, g, d = f.eval(), g.eval(), d.eval()
    f_copy, g, d = f_copy.to(device), g.to(device), d.to(device)
    init_img = test_data[idx][0].unsqueeze(0)

    init_shape = init_img.shape
    img = init_img[0, 0].cpu().data.numpy()

    # Complete the computations
    init_img=init_img.to(device)
    output, inter = f_copy(init_img)
    embed1 = g(inter)
    weights = h.fc1.weight.cpu().data.numpy()
    pred = h(embed1).argmax(dim=1).cpu().data.numpy()
    print ("Predicted class:", classes[pred[0]])
    expl1 = embed1[0].cpu().data.numpy() * weights[pred[0]]
    expl1 = expl1 / np.abs(expl1).max()
    attr_idx1 = expl1.argsort()[-1]
    attr_idx2 = expl1.argsort()[-2]
    attr_idx3 = expl1.argsort()[-3]
    cur_img1 = optimize_inp(f_copy, g, attr_idx1, device, list(init_img.shape), init=0.2*init_img, lmbd_bound=10.0, lmbd_tv=6.0, C=2.0, lmbd_l1=0.0)
    cur_img2 = optimize_inp(f_copy, g, attr_idx2, device, list(init_img.shape), init=0.2*init_img, lmbd_bound=10.0, lmbd_tv=6.0, C=2.0, lmbd_l1=0.0)
    cur_img3 = optimize_inp(f_copy, g, attr_idx3, device, list(init_img.shape), init=0.2*init_img, lmbd_bound=10.0, lmbd_tv=6.0, C=2.0, lmbd_l1=0.0)

    # Make the plot
    fig = plt.figure(figsize=(12, 5))
    fig.add_subplot(1, 2, 1)
    plt.imshow(img)
    #plt.title('Input sample', fontsize=28)
    plt.axis('off')

    fig.add_subplot(1, 2, 2)
    x_pos = np.array(range(expl1.shape[0])).astype(str)
    x_pos = np.array(['$\phi_{'+i+'}$' for i in x_pos])
    plt.bar(list(range(n_attr)), -1*np.sort(-1*expl1)[:n_attr], color='green')
    plt.xticks(list(range(n_attr)), x_pos[np.array([attr_idx1, attr_idx2, attr_idx3])], fontsize=32)
    plt.yticks([0, 0.5, 1], [0.0, 0.5, 1.0], fontsize=23)
    plt.ylabel('Relevance to prediction', fontsize=24)
    #plt.subplots_adjust(wspace=0.05)
    if not save:
        plt.show()
    else:
        plt.savefig(location + '/s' + str(idx) + '_rel_')
        plt.close()

    fig = plt.figure(figsize=(12, 5))
    fig.add_subplot(1, 3, 1)
    plt.imshow(cur_img1[0, 0])
    plt.axis('off')
    plt.title('$\phi_{'+str(attr_idx1)+'}$', fontsize=32)
    fig.add_subplot(1, 3, 2)
    plt.imshow(cur_img2[0, 0])
    plt.axis('off')
    plt.title('$\phi_{'+str(attr_idx2)+'}$', fontsize=32)
    fig.add_subplot(1, 3, 3)
    plt.imshow(cur_img3[0, 0])
    plt.axis('off')
    plt.title('$\phi_{'+str(attr_idx3)+'}$', fontsize=32)
    plt.subplots_adjust(wspace=0.05)
    if not save:
        plt.show()
    else:
        plt.savefig(location + '/s' + str(idx) + '_att3_')
        plt.close()
 
    return cur_img1, cur_img2