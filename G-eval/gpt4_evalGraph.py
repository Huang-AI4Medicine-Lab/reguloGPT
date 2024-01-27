import openai
import json
import argparse
import time
import pandas as pd
import math
def load_file(finalename):
    df = pd.read_csv(finalename, encoding='ISO-8859-1')

    res = []
    edges = []
    ctxLst = []
    ctx = None
    cnt = 0
    idxLst = []
    for index, row in df.iterrows():

        if not math.isnan(row['Idx']):
            ctx = (row['Title'], row['Context'])
            edges.append( "(" + row['node-A'].strip().lower() + ", " +  row['edge'].strip().lower() + ", " + row['node-B'].strip().lower() + ")")
            # row['node-A'].strip().lower() + " " + row['edge'].strip().lower() + " " + row['node-B'].strip().lower())

            if int(row['Idx']) not in idxLst:
                idxLst.append(int(row['Idx']))
            # Set.add(int(row['index']))
        elif len(edges) > 0:
            cnt += 1
            ctxLst.append(ctx)
            res.append(edges)
            edges = []


    cnt += 1
    ctxLst.append(ctx)
    res.append(edges)

    return res, ctxLst, idxLst



if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    # argparser.add_argument('--key', type=str, required=True)
    argparser.add_argument('--key', type=str, default='ss')
    argparser.add_argument('--model', type=str, default='gpt-4')
    args = argparser.parse_args()
    # openai.api_key = args.key
    openai.api_key = 'API_KEY'

    # evalType = 1
    evalType = 2

    # pr_file = "baseline_0_1_edge.csv"
    # pr_file = "baseline_0_3_edge.csv"

    # pr_file = "baseline_1_4_edge.csv"
    
    pr_file = "baseline_2_4_edge.csv"
    print("graph/" + str(evalType) + "/evaluation.txt", pr_file)

    preLst, ctx_pr, pr_idx = load_file(finalename="../data/" + pr_file) 
    prompt = open("graph/" + str(evalType) + "/evaluation.txt").read()
  
    ####################
    # cur_prompt = prompt
    # while True:
    #     try:
    #         # https://platform.openai.com/docs/api-reference/completions/create
    #         _response = openai.ChatCompletion.create(
    #             model=args.model,
    #             messages=[{"role": "system", "content": cur_prompt}],
    #             temperature=0,
    #             n=1
    #         )
    #         time.sleep(0.5)
    #         # print("=======================")
    #         # print("cur_prompt \n", cur_prompt)

    #         new_output = [_response['choices'][i]['message']['content'] for i in
    #                          range(len(_response['choices']))]
    #         # print("=======================")
    #         # print(new_sentences)
    #         # ct += 1
    #         break
    #     except Exception as e:
    #         print(">>>>>>>>>>>>>>>>>>>>")

    #         print(e)
    #         if ("limit" in str(e)):
    #             time.sleep(2)
    #         else:
    #             ignore += 1
    #             print('ignored', ignore)

    #             break
    # print(new_output[0])

    ########################
    ct, ignore = 0, 0
    new_json   = []
    cur_scores = []
    for i, (edges, ctx) in enumerate(zip(preLst, ctx_pr)):
        print(i)
        title = ctx[0].strip().lower() + "."
        context = ctx[1].strip().lower()

        if evalType == 0:
            graph = "Context: " + context + "\nGraph: " + ', '.join(edges)
        elif evalType == 1:
            graph = ', '.join(edges) 
        elif evalType == 2:
            graph = "Context: " + context
        # print(graph)
        cur_prompt = prompt.replace('{{Sentence}}', title).replace('{{Graph}}', graph)

        # print(cur_prompt)
        # break
        while True:
            try:
                # https://platform.openai.com/docs/api-reference/completions/create
                _response = openai.ChatCompletion.create(
                    model=args.model,
                    messages=[{"role": "system", "content": cur_prompt}],
                    temperature=2,
                    max_tokens=5,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0,
                    stop=None,
                    # logprobs=40,
                    n=5
                )
                time.sleep(0.5)
                # print("=======================")
                # print("cur_prompt \n", cur_prompt)

                cur_scores = [_response['choices'][i]['message']['content'] for i in
                                 range(len(_response['choices']))]
                try:
                    res = [float(st) for st in cur_scores]
                    break
                except Exception as e:
                    print(i)
                # ct += 1
            except Exception as e:
                print(">>>>>>>>>>>>>>>>>>>>")

                print(e)
                if ("limit" in str(e)):
                    time.sleep(2)
                else:
                    ignore += 1
                    print('ignored', ignore)

                    break
        
        # print(cur_scores)
        res = {'idx':pr_idx[i], 'title': title, 'score': cur_scores}
        new_json.append(res)

        with open('graph/' + pr_file[:-3] + 'json', 'w') as f:
            json.dump(new_json, f, indent=4)

