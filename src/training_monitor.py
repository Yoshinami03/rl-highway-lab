"""
学習品質監視モジュール

PPOの学習中にメトリクスを監視し、品質基準を満たさない場合に学習を中断します。
"""

from stable_baselines3.common.callbacks import BaseCallback
import numpy as np


class TrainingQualityMonitor(BaseCallback):
    """
    学習品質を監視し、基準を満たさない場合に学習を中断するコールバック
    
    評価基準:
    1. explained_variance: 0.1以上に向かっている（改善傾向）
    2. approx_kl: 0.005～0.02の範囲内
    3. clip_fraction: 0.05以上
    
    最初のmax_checks回のチェックのみ実施し、1回でも2/3以上の条件を満たせば監視を終了します。
    max_checks回すべて失敗した場合は学習を中断します。
    """
    
    def __init__(
        self,
        check_freq: int = 10,  # チェック頻度（イテレーション）
        min_explained_variance: float = 0.1,
        approx_kl_range: tuple = (0.005, 0.02),
        min_clip_fraction: float = 0.05,
        max_checks: int = 4,  # 最大チェック回数
        verbose: int = 1
    ):
        """
        Args:
            check_freq: チェック頻度（イテレーション数）
            min_explained_variance: explained_varianceの最小目標値
            approx_kl_range: approx_klの許容範囲 (min, max)
            min_clip_fraction: clip_fractionの最小目標値
            max_checks: 最大チェック回数（この回数だけチェックを実施）
            verbose: ログ出力レベル（0: なし, 1: 詳細）
        """
        super().__init__(verbose)
        self.check_freq = check_freq
        self.min_explained_variance = min_explained_variance
        self.approx_kl_range = approx_kl_range
        self.min_clip_fraction = min_clip_fraction
        self.max_checks = max_checks
        self.verbose = verbose
        
        # 履歴管理
        self.explained_variance_history = []
        self.check_count = 0  # 実施したチェック回数
        self.monitoring_active = True  # 監視が有効かどうか
        self.iteration_count = 0
    
    def _on_step(self) -> bool:
        """
        各ステップで呼ばれる（ここでは何もしない）
        
        Returns:
            bool: True（学習を継続）
        """
        return True
    
    def _on_rollout_end(self) -> bool:
        """
        ロールアウト終了時に品質をチェック
        
        Returns:
            bool: True（学習を継続）、False（学習を中断）
        """
        self.iteration_count += 1
        
        # 監視が無効化されている場合はスキップ
        if not self.monitoring_active:
            return True
        
        # チェック頻度に達していない場合はスキップ
        if self.iteration_count % self.check_freq != 0:
            return True
        
        # チェック回数をインクリメント
        self.check_count += 1
        
        # メトリクスを取得
        try:
            explained_var = self.logger.name_to_value.get("train/explained_variance", None)
            approx_kl = self.logger.name_to_value.get("train/approx_kl", None)
            clip_fraction = self.logger.name_to_value.get("train/clip_fraction", None)
            
            if None in [explained_var, approx_kl, clip_fraction]:
                return True  # メトリクスが取得できない場合は続行
            
            # 条件チェック
            conditions_met = 0
            
            # 1. explained_variance が改善傾向か
            self.explained_variance_history.append(explained_var)
            if len(self.explained_variance_history) >= 2:
                # 最近の値が増加傾向、または既に基準を満たしている
                if explained_var >= self.min_explained_variance:
                    conditions_met += 1
                elif explained_var > self.explained_variance_history[-2]:
                    conditions_met += 1
            else:
                # 初回は基準値と比較
                if explained_var >= self.min_explained_variance:
                    conditions_met += 1
            
            # 2. approx_kl が範囲内か
            if self.approx_kl_range[0] <= approx_kl <= self.approx_kl_range[1]:
                conditions_met += 1
            
            # 3. clip_fraction が基準以上か
            if clip_fraction >= self.min_clip_fraction:
                conditions_met += 1
            
            # 結果を出力
            if self.verbose >= 1:
                print(f"\n=== Training Quality Check (Check {self.check_count}/{self.max_checks}, Iteration {self.iteration_count}) ===")
                print(f"Explained Variance: {explained_var:.4f} (target: >= {self.min_explained_variance})")
                print(f"Approx KL: {approx_kl:.6f} (target: {self.approx_kl_range[0]}-{self.approx_kl_range[1]})")
                print(f"Clip Fraction: {clip_fraction:.4f} (target: >= {self.min_clip_fraction})")
                print(f"Conditions met: {conditions_met}/3")
            
            # 2つ以上の条件を満たしているかチェック
            if conditions_met >= 2:
                # 合格 → 監視終了
                self.monitoring_active = False
                if self.verbose >= 1:
                    print(f"✓ Quality check passed! Monitoring disabled, training will continue.")
                return True
            else:
                # 不合格
                if self.verbose >= 1:
                    print(f"⚠️  Warning: Only {conditions_met}/3 conditions met.")
                
                # 最大チェック回数に達した場合は学習中断
                if self.check_count >= self.max_checks:
                    if self.verbose >= 1:
                        print(f"\n❌ Training stopped: Failed all {self.max_checks} quality checks.")
                    return False  # 学習を中断
            
        except Exception as e:
            if self.verbose >= 1:
                print(f"Error in quality check: {e}")
        
        return True  # 学習を継続

