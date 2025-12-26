"""
Verify all "SOTA" with the same tokenization and BLEU script.
"""

import os
import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import sacrebleu
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FairEvaluator:
    """公平评估器 - 标准化所有SOTA模型的评估"""
    
    def __init__(self, 
                 tokenizer_name: str = "bert-base-uncased",
                 bleu_version: str = "2.3.1",
                 case_sensitive: bool = False,
                 tokenize_method: str = "13a",
                 output_dir: str = "./eval_results"):
        """
        初始化公平评估器
        
        Args:
            tokenizer_name: 用于tokenization的tokenizer
            bleu_version: sacreBLEU版本
            case_sensitive: 是否区分大小写 (SLT通常为False)
            tokenize_method: tokenization方法 ('13a', 'moses', 'intl')
            output_dir: 输出目录
        """
        self.tokenizer_name = tokenizer_name
        self.bleu_version = bleu_version
        self.case_sensitive = case_sensitive
        self.tokenize_method = tokenize_method
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # 创建sacreBLEU对象
        self.bleu = sacrebleu.BLEU(
            tokenize=tokenize_method,
            lowercase=not case_sensitive,
            version=bleu_version
        )
        
        # 记录评估配置
        self.config = {
            "tokenizer_name": tokenizer_name,
            "bleu_version": bleu_version,
            "case_sensitive": case_sensitive,
            "tokenize_method": tokenize_method,
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
        logger.info(f"初始化公平评估器: {self.config}")
    
    def load_reference(self, ref_path: str) -> List[str]:
        """加载参考翻译"""
        with open(ref_path, 'r', encoding='utf-8') as f:
            references = [line.strip() for line in f if line.strip()]
        logger.info(f"加载参考翻译: {len(references)} 句")
        return references
    
    def load_hypothesis(self, hyp_path: str) -> List[str]:
        """加载模型输出"""
        with open(hyp_path, 'r', encoding='utf-8') as f:
            hypotheses = [line.strip() for line in f if line.strip()]
        logger.info(f"加载模型输出: {len(hypotheses)} 句")
        return hypotheses
    
    def standardize_tokenization(self, texts: List[str]) -> List[str]:
        """标准化tokenization - 确保所有模型使用相同处理"""
        # 使用预训练tokenizer进行标准化
        tokenized = []
        for text in texts:
            # 移除多余空格，标准化标点
            text = ' '.join(text.split())  # 标准化空格
            # 可选：使用tokenizer进行subword-level标准化
            # tokens = self.tokenizer.tokenize(text)
            # text = self.tokenizer.convert_tokens_to_string(tokens)
            tokenized.append(text)
        return tokenized
    
    def evaluate_bleu(self, 
                     hypotheses: List[str], 
                     references: List[str]) -> Dict:
        """计算BLEU分数"""
        # 标准化输入
        hyp_std = self.standardize_tokenization(hypotheses)
        refs_std = [[self.standardize_tokenization([ref])[0] for ref in ref_group] 
                   for ref_group in references]
        
        # 计算BLEU
        bleu_score = self.bleu.corpus_bleu(hyp_std, refs_std)
        
        result = {
            "bleu": bleu_score.score,
            "bleu1": bleu_score.precisions[0],
            "bleu2": bleu_score.precisions[1],
            "bleu3": bleu_score.precisions[2],
            "bleu4": bleu_score.precisions[3],
            "bp": bleu_score.bp,
            "length_ratio": bleu_score.sys_len / bleu_score.ref_len,
            "signature": bleu_score.format(signature=True),
            "counts": bleu_score.counts,
            "totals": bleu_score.totals
        }
        
        logger.info(f"BLEU: {result['bleu']:.2f} | Signature: {result['signature']}")
        return result
    
    def evaluate_chrf(self, hypotheses: List[str], references: List[str]) -> Dict:
        """计算chrF分数"""
        hyp_std = self.standardize_tokenization(hypotheses)
        refs_std = [[self.standardize_tokenization([ref])[0] for ref in ref_group] 
                   for ref_group in references]
        
        chrf = sacrebleu.corpus_chrf(hyp_std, refs_std)
        return {
            "chrf": chrf.score,
            "chrf2": chrf.score_beta2 if hasattr(chrf, 'score_beta2') else None
        }
    
    def save_results(self, 
                    model_name: str, 
                    bleu_results: Dict, 
                    chrf_results: Dict = None,
                    hypotheses: List[str] = None,
                    save_hyp: bool = True):
        """保存评估结果"""
        results = {
            "model_name": model_name,
            "config": self.config,
            "bleu": bleu_results,
            "chrf": chrf_results if chrf_results else {}
        }
        
        # 保存JSON结果
        result_file = self.output_dir / f"{model_name}_results.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # 保存标准化后的hypotheses
        if save_hyp and hypotheses:
            hyp_file = self.output_dir / f"{model_name}_hypotheses.txt"
            with open(hyp_file, 'w', encoding='utf-8') as f:
                for hyp in hypotheses:
                    f.write(hyp + '\n')
        
        logger.info(f"结果已保存: {result_file}")
        return result_file
    
    def compare_models(self, 
                      model_results: List[Dict]) -> pd.DataFrame:
        """比较多个模型的结果"""
        comparison_data = []
        for result in model_results:
            row = {
                "model": result["model_name"],
                "bleu": result["bleu"]["bleu"],
                "bleu1": result["bleu"]["bleu1"],
                "bleu2": result["bleu"]["bleu2"],
                "bleu3": result["bleu"]["bleu3"],
                "bleu4": result["bleu"]["bleu4"],
                "bp": result["bleu"]["bp"],
                "chrf": result.get("chrf", {}).get("chrf", None),
                "signature": result["bleu"]["signature"]
            }
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        # 排序
        df = df.sort_values("bleu", ascending=False)
        
        # 保存比较表
        comparison_file = self.output_dir / "model_comparison.csv"
        df.to_csv(comparison_file, index=False, encoding='utf-8')
        
        # 保存LaTeX表格（用于论文）
        latex_file = self.output_dir / "model_comparison.tex"
        with open(latex_file, 'w', encoding='utf-8') as f:
            f.write(df.to_latex(index=False, float_format="%.2f", 
                              column_format="lccccccl", escape=False))
        
        logger.info(f"比较结果已保存: {comparison_file}")
        return df

def load_sota_models_config(config_path: str) -> List[Dict]:
    """加载SOTA模型配置"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config["models"]

def main():
    parser = argparse.ArgumentParser(description="公平评估SOTA模型")
    parser.add_argument("--ref_path", required=True, help="参考翻译文件路径")
    parser.add_argument("--model_configs", required=True, help="模型配置JSON路径")
    parser.add_argument("--output_dir", default="./eval_results", help="输出目录")
    parser.add_argument("--case_insensitive", action="store_true", 
                       help="使用case-insensitive评估 (SLT标准)")
    parser.add_argument("--tokenizer", default="bert-base-uncased", 
                       help="用于标准化的tokenizer")
    parser.add_argument("--test_set", default="wmt14", 
                       help="测试集名称")
    
    args = parser.parse_args()
    
    # 初始化评估器
    evaluator = FairEvaluator(
        tokenizer_name=args.tokenizer,
        case_sensitive=not args.case_insensitive,
        output_dir=args.output_dir
    )
    
    # 加载参考翻译
    references = evaluator.load_reference(args.ref_path)
    
    # 加载模型配置
    model_configs = load_sota_models_config(args.model_configs)
    
    # 评估所有模型
    all_results = []
    for model_config in model_configs:
        model_name = model_config["name"]
        hyp_path = model_config["hypothesis_path"]
        
        logger.info(f"评估模型: {model_name}")
        
        # 加载模型输出
        hypotheses = evaluator.load_hypothesis(hyp_path)
        
        # 确保句子数量匹配
        min_len = min(len(hypotheses), len(references))
        hypotheses = hypotheses[:min_len]
        references = references[:min_len]
        
        # 计算BLEU
        bleu_results = evaluator.evaluate_bleu(hypotheses, [references])
        
        # 计算chrF
        chrf_results = evaluator.evaluate_chrf(hypotheses, [references])
        
        # 保存结果
        result_file = evaluator.save_results(
            model_name, bleu_results, chrf_results, hypotheses
        )
        
        # 收集结果用于比较
        all_results.append({
            "model_name": model_name,
            "bleu": bleu_results,
            "chrf": chrf_results,
            "result_file": str(result_file)
        })
    
    # 比较所有模型
    comparison_df = evaluator.compare_models(all_results)
    print("\n=== SOTA模型公平比较结果 ===")
    print(comparison_df.to_string(index=False))
    
    # 保存配置
    config_summary = {
        "evaluation_config": evaluator.config,
        "models_evaluated": [r["model_name"] for r in all_results],
        "comparison_table": comparison_df.to_dict('records')
    }
    
    config_file = Path(args.output_dir) / "evaluation_summary.json"
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config_summary, f, ensure_ascii=False, indent=2)
    
    logger.info(f"评估完成! 总结文件: {config_file}")

if __name__ == "__main__":
    main()