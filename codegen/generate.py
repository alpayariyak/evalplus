import asyncio
import argparse
import os
from os import PathLike
from dotenv import load_dotenv
from model import DecoderBase, make_model
import tqdm
import logging

def construct_contract_prompt(prompt: str, contract_type: str, contract: str) -> str:
    if contract_type == "none":
        return prompt
    elif contract_type == "docstring":
        # embed within the docstring
        sep = ""
        if '"""' in prompt:
            sep = '"""'
        elif "'''" in prompt:
            sep = "'''"
        assert sep != ""
        l = prompt.split(sep)
        contract = "\n".join([x.split("#")[0] for x in contract.splitlines()])
        l[1] = (
            l[1] + contract + "\n" + " " * (len(contract) - len(contract.lstrip()) - 1)
        )
        return sep.join(l)
    elif contract_type == "code":
        # at the beginning of the function
        contract = "\n".join([x.split("#")[0] for x in contract.splitlines()])
        return prompt + contract


async def code_generate(args, workdir: PathLike, model: DecoderBase, id_range=None, id_list=None, parallel_reqs=1):
    if args.dataset == "humaneval":
        from evalplus.data import get_human_eval_plus

        dataset = get_human_eval_plus()
    elif args.dataset == "mbpp":
        from evalplus.data import get_mbpp_plus

        dataset = get_mbpp_plus()

    valid_tasks = []
    for task_id, task in dataset.items():
        id_num = int(task_id.split("/")[1])
        if id_range is not None:
            if id_list is not None:
                raise ValueError("id_range and id_list cannot be used together")
            low, high = id_range
            if id_num < low or id_num >= high:
                logging.info(f"Skipping {task_id} as it is not in {id_range}")
                continue
        
        if id_list is not None:
            if id_num not in id_list:
                logging.info(f"Skipping {task_id} as it is not in {id_list}")
                continue
        valid_tasks.append((task_id, task))

    async def handle_task(task_id, task):
        p_name = task_id.replace("/", "_")
        if args.contract_type != "none" and task["contract"] == "":
            return
        os.makedirs(os.path.join(workdir, p_name), exist_ok=True)
        log = f"Codegen: {p_name} @ {model}"
        n_existing = 0
        if args.resume:
            # count existing .py files
            n_existing = len(
                [
                    f
                    for f in os.listdir(os.path.join(workdir, p_name))
                    if f.endswith(".py")
                ]
            )
            if n_existing > 0:
                log += f" (resuming from {n_existing})"

        nsamples = args.n_samples - n_existing
        logging.info(log)

        sidx = args.n_samples - nsamples
        while sidx < args.n_samples:
            model.dataset = args.dataset
            outputs = await model.codegen(
                construct_contract_prompt(
                    task["prompt"], args.contract_type, task["contract"]
                ).strip(),
                do_sample=not args.greedy,
                num_samples=args.n_samples - sidx,
            )
            assert outputs, "No outputs from model!"
            for impl in outputs:
                try:
                    with open(
                        os.path.join(workdir, p_name, f"{sidx}.py"),
                        "w",
                        encoding="utf-8",
                    ) as f:
                        if model.direct_completion:
                            f.write(task["prompt"] + impl)
                        else:
                            f.write(impl)
                except UnicodeEncodeError:
                    continue
                sidx += 1
    
    batches_inputs = [valid_tasks[i:i + parallel_reqs] for i in range(0, len(valid_tasks), parallel_reqs)]
    
    for batch in tqdm.tqdm(batches_inputs, total=len(batches_inputs), desc="Generating code in batches"):
        await asyncio.gather(*[handle_task(task_id, task) for task_id, task in batch])
   

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument("--bs", default=1, type=int)
    parser.add_argument("--temperature", default=0.0, type=float)
    parser.add_argument(
        "--dataset", required=True, type=str, choices=["humaneval", "mbpp"]
    )
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--n_samples", default=1, type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--contract-type",
        default="none",
        type=str,
        choices=["none", "code", "docstring"],
    )
    parser.add_argument("--greedy", action="store_true")
    # id_range is list
    parser.add_argument("--id-range", default=None, nargs="+", type=int)
    parser.add_argument("--id-list", default=None, nargs="+", type=int)
    parser.add_argument("--parallel", default=1, type=int, help="Number of parallel requests to make to the model")
    args = parser.parse_args()

    if args.greedy and (args.temperature != 0 or args.bs != 1 or args.n_samples != 1):
        args.temperature = 0
        args.bs = 1
        args.n_samples = 1
        print("Greedy decoding ON (--greedy): setting bs=1, n_samples=1, temperature=0")

    if args.id_range is not None:
        assert len(args.id_range) == 2, "id_range must be a list of length 2"
        assert args.id_range[0] < args.id_range[1], "id_range must be increasing"
        args.id_range = tuple(args.id_range)

    #Load environment variables
    load_dotenv()
    
    # Make project dir
    os.makedirs(args.root, exist_ok=True)
    # Make dataset dir
    os.makedirs(os.path.join(args.root, args.dataset), exist_ok=True)
    # Make dir for codes generated by each model
    args.model = args.model
    model = make_model(
        name=args.model,
        batch_size=args.bs,
        temperature=args.temperature,
        dataset=args.dataset,
    )
    workdir = os.path.join(
        args.root,
        args.dataset,
        args.model
        + f"_temp_{args.temperature}"
        + ("" if args.contract_type == "none" else f"-contract-{args.contract_type}"),
    )
    os.makedirs(workdir, exist_ok=True)

    with open(os.path.join(workdir, "args.txt"), "w") as f:
        f.write(str(args))

    await code_generate(args, workdir=workdir, model=model, id_range=args.id_range, id_list=args.id_list, parallel_reqs=args.parallel)


if __name__ == "__main__":
    asyncio.run(main())
