import torch
import torch.nn.functional as F
import time
import os, pdb
import json
from core.utils import AverageMeter, calculate_accuracy


def calculate_video_results(output_buffer, video_id, test_results, class_names):
    video_outputs = torch.stack(output_buffer)
    average_scores = torch.mean(video_outputs, dim=0)
    sorted_scores, locs = torch.topk(average_scores, k=10)

    video_results = []
    for i in range(sorted_scores.size(0)):
        video_results.append({
            'label': class_names[locs[i]],
            'score': sorted_scores[i].cpu().item()
        })

    test_results['results_per_video'][video_id.item()] = video_results


def test(logger, data_loader, model, opt, class_names):
    print('test')

    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()

    end_time = time.time()
    output_buffer = []
    previous_video_id = ''

    collect_outputs = list()
    collect_targets = list()
    test_results = {'results_per_video': {}, 'results_overall': {}}
    for i, (inputs, targets) in enumerate(data_loader):
        data_time.update(time.time() - end_time)

        if isinstance(inputs, torch.Tensor):
            inputs = inputs.cuda()
        else:
            inputs = [i.cuda() for i in inputs]
        targets = targets.cuda()

        # inputs = Variable(inputs, volatile=True)
        with torch.no_grad():
            outputs = model(inputs)

        if not opt.no_softmax_in_test:
            outputs = F.softmax(outputs)

        collect_outputs.append(outputs.cpu())
        collect_targets.append(targets.cpu())

        targets = targets.cpu()
        for j in range(outputs.size(0)):
            if not (i == 0 and j == 0) and targets[j] != previous_video_id:
                calculate_video_results(output_buffer, previous_video_id,
                                        test_results, class_names)
                output_buffer = []
            output_buffer.append(outputs[j].data.cpu())
            previous_video_id = targets[j]

        if (i % 100) == 0:
            with open(
                    os.path.join(opt.output_dir, 'test_{}_{}_{}.json'.format(
                        opt.test_dataset, opt.test_perturbation, opt.test_severity)), 'w') as f:
                json.dump(test_results, f)


        batch_time.update(time.time() - end_time)
        end_time = time.time()

        print('[{}/{}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'.format(
                  i + 1,
                  len(data_loader),
                  batch_time=batch_time,
                  data_time=data_time))

    outputs = torch.cat(collect_outputs)
    targets = torch.cat(collect_targets)
    accuracy = calculate_accuracy(outputs, targets)

    test_results['results_overall'] = accuracy
    logger.info(f"Final accuracy is {round(accuracy, 4)}")

    with open(
            os.path.join(opt.output_dir, 'test_{}_{}_{}.json'.format(opt.test_dataset, opt.test_perturbation,
                                                                     opt.test_severity)),
            'w') as f:
        json.dump(test_results, f)
