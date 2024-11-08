import requests
import time
from datetime import datetime
from typing import List, Dict, Tuple
import joblib
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import json
import logging
import os

# Configuration des logs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('swap_analyzer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SolanaSwapAnalyzer:
    def __init__(self, rpc_url: str):
        logger.info("Initialisation de SolanaSwapAnalyzer...")
        self.rpc_url = rpc_url
        self.last_request_time = 0
        self.request_count = 0
        self.RATE_LIMIT = {
            'requests_per_10s': 40,
            'min_interval': 0.25
        }
        
        # Liste des programmes de swap connus
        self.SWAP_PROGRAMS = {
            "JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4": "Jupiter",
            "whirLbMiicVdio4qvUfM5KAg6Ct8VwpYzGff3uctyCc": "Orca",
            "9W959DqEETiGZocYWCQPaJ6sBmUzgfxXfqGeTEdp3aQP": "Raydium",
            "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8": "Raydium V4",
            "SwaPpA9LAaLfeLi3a68M4DjnLqgtticKg6CnyNwgAC8": "Raydium Legacy",
            "DjVE6JNiYqPL2QXyCUUh8rNjHrbz9hXHNYt99MQ59qw1": "Orca Whirlpool",
        }
        
        try:
            logger.info("Chargement du modèle de classification...")
            self.model = joblib.load('models/wallet_classifier_pipeline.joblib')
            logger.info("Modèle chargé avec succès")
        except Exception as e:
            logger.error(f"Erreur lors du chargement du modèle: {e}")
            self.model = None

    def _make_rpc_request(self, method: str, params: List) -> Dict:
        """Effectue une requête RPC avec gestion du rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        # Gestion du rate limiting
        if time_since_last < self.RATE_LIMIT['min_interval']:
            sleep_time = self.RATE_LIMIT['min_interval'] - time_since_last
            logger.debug(f"Rate limiting: pause de {sleep_time:.2f}s")
            time.sleep(sleep_time)
        
        logger.debug(f"Envoi requête RPC: {method} {params}")
        headers = {"Content-Type": "application/json"}
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params
        }
        
        try:
            response = requests.post(self.rpc_url, headers=headers, json=payload)
            response.raise_for_status()
            self.last_request_time = time.time()
            return response.json()
        except Exception as e:
            logger.error(f"Erreur RPC {method}: {e}")
            return {"error": str(e)}

    def _analyze_token_changes(self, transaction: Dict) -> Tuple[List[Tuple[float, str]], List[Tuple[float, str]]]:
        """Analyse les changements de balance de tokens dans une transaction"""
        logger.debug("Analyse des changements de tokens...")
        
        pre_balances = {
            b['mint']: {
                'amount': float(b.get('uiTokenAmount', {}).get('uiAmount', 0) or 0),
                'symbol': b.get('uiTokenAmount', {}).get('symbol', '')
            }
            for b in transaction.get('meta', {}).get('preTokenBalances', [])
        }
        
        post_balances = {
            b['mint']: {
                'amount': float(b.get('uiTokenAmount', {}).get('uiAmount', 0) or 0),
                'symbol': b.get('uiTokenAmount', {}).get('symbol', '')
            }
            for b in transaction.get('meta', {}).get('postTokenBalances', [])
        }
        
        logger.debug(f"Balances pré-transaction: {pre_balances}")
        logger.debug(f"Balances post-transaction: {post_balances}")
        
        tokens_in = []
        tokens_out = []
        
        all_mints = set(list(pre_balances.keys()) + list(post_balances.keys()))
        
        for mint in all_mints:
            pre_amount = pre_balances.get(mint, {'amount': 0})['amount']
            post_amount = post_balances.get(mint, {'amount': 0})['amount']
            symbol = post_balances.get(mint, {'symbol': mint[:8]})['symbol']
            
            diff = post_amount - pre_amount
            if abs(diff) > 0.000001:  # Seuil minimal pour éviter le bruit
                if diff > 0:
                    tokens_in.append((diff, symbol))
                    logger.debug(f"Token entrant: {diff} {symbol}")
                else:
                    tokens_out.append((abs(diff), symbol))
                    logger.debug(f"Token sortant: {abs(diff)} {symbol}")
        
        return tokens_in, tokens_out

    def _is_swap_transaction(self, transaction: Dict) -> Tuple[bool, str]:
        """Vérifie si une transaction est un swap et retourne le protocole"""
        logger.debug("Vérification si la transaction est un swap...")
        
        message = transaction.get('transaction', {}).get('message', {})
        instructions = message.get('instructions', [])
        logs = transaction.get('meta', {}).get('logMessages', [])
        
        # Vérification dans les instructions
        for instruction in instructions:
            program_id = instruction.get('programId')
            if program_id in self.SWAP_PROGRAMS:
                protocol = self.SWAP_PROGRAMS[program_id]
                logger.info(f"Swap détecté - Protocol: {protocol}")
                return True, protocol
        
        # Vérification dans les logs
        if logs:
            for program_id, protocol in self.SWAP_PROGRAMS.items():
                if any(program_id in log for log in logs):
                    logger.info(f"Swap détecté dans les logs - Protocol: {protocol}")
                    return True, protocol
        
        logger.debug("Aucun swap détecté")
        return False, ""

    def _extract_features(self, transactions: List[Dict]) -> Dict:
        """Extrait les caractéristiques des transactions pour la classification"""
        logger.info("Extraction des features pour la classification...")
        
        # Dictionnaire pour stocker les features extraites
        raw_features = {}
        
        # Caractéristiques temporelles de base
        timestamps = [tx.get('blockTime', 0) for tx in transactions]
        if len(timestamps) > 1:
            time_diffs = np.diff(timestamps)
            raw_features['avg_time_between_tx'] = np.mean(time_diffs)
            raw_features['time_variance'] = np.std(time_diffs)
        else:
            raw_features['avg_time_between_tx'] = 0
            raw_features['time_variance'] = 0
        
        logger.debug(f"Features temporelles extraites")
        
        # Caractéristiques des transactions
        raw_features['total_transactions'] = len(transactions)
        
        # Analyse des frais
        fees = [tx.get('meta', {}).get('fee', 0) for tx in transactions]
        raw_features['avg_fee'] = np.mean(fees) if fees else 0
        raw_features['std_fee'] = np.std(fees) if fees else 0
        
        # Analyse des instructions
        instruction_counts = [len(tx.get('transaction', {}).get('message', {}).get('instructions', [])) 
                            for tx in transactions]
        raw_features['avg_instructions_per_tx'] = np.mean(instruction_counts) if instruction_counts else 0
        raw_features['instruction_complexity_score'] = np.std(instruction_counts) if instruction_counts else 0
        
        # Calcul du temps moyen entre transactions pour chaque compte
        account_time_pairs = defaultdict(list)
        for tx in transactions:
            accounts = []
            message = tx.get('transaction', {}).get('message', {})
            if 'accountKeys' in message:
                for acc in message['accountKeys']:
                    if isinstance(acc, dict):
                        acc_key = acc.get('pubkey', '')
                    else:
                        acc_key = acc
                    accounts.append(acc_key)
            
            timestamp = tx.get('blockTime', 0)
            for account in accounts:
                account_time_pairs[account].append(timestamp)
        
        account_avg_times = []
        for timestamps in account_time_pairs.values():
            if len(timestamps) > 1:
                times_sorted = sorted(timestamps)
                time_diffs = np.diff(times_sorted)
                if len(time_diffs) > 0:
                    account_avg_times.append(np.mean(time_diffs))
        
        raw_features['avg_time_between_account_tx'] = np.mean(account_avg_times) if account_avg_times else 0
        
        # Diversité des comptes
        all_accounts = []
        for tx in transactions:
            message = tx.get('transaction', {}).get('message', {})
            accounts = message.get('accountKeys', [])
            all_accounts.extend([acc.get('pubkey', '') if isinstance(acc, dict) else acc for acc in accounts])
        
        unique_accounts = set(all_accounts)
        raw_features['unique_accounts_count'] = len(unique_accounts)
        raw_features['account_diversity_score'] = len(unique_accounts) / len(all_accounts) if all_accounts else 0
        
        # Interactions avec le programme système
        system_program = '11111111111111111111111111111111'
        system_interactions = sum(1 for tx in transactions 
                                if system_program in [acc.get('pubkey', '') if isinstance(acc, dict) else acc 
                                                    for acc in tx.get('transaction', {}).get('message', {}).get('accountKeys', [])])
        raw_features['system_program_interaction_ratio'] = system_interactions / len(transactions) if transactions else 0
        
        # Extraction des signatures
        signatures = []
        for tx in transactions:
            tx_signatures = tx.get('transaction', {}).get('signatures', [])
            signatures.extend(tx_signatures)
        
        raw_features['signature_entropy'] = self._calculate_entropy(signatures) if signatures else 0
        
        # Variations de slots
        slots = [tx.get('slot', 0) for tx in transactions]
        raw_features['slot_variation'] = np.std(slots) if slots else 0
        
        # Liste ordonnée des features comme dans le modèle entraîné
        ordered_feature_names = [
            'total_transactions',
            'avg_fee',
            'std_fee',
            'avg_time_between_tx',
            'time_variance',
            'avg_time_between_account_tx',
            'unique_accounts_count',
            'account_diversity_score',
            'avg_instructions_per_tx',
            'instruction_complexity_score',
            'system_program_interaction_ratio',
            'signature_entropy',
            'slot_variation'
        ]
        
        # Création du DataFrame avec l'ordre correct des features
        features_ordered = {name: raw_features.get(name, 0) for name in ordered_feature_names}
        
        # Vérification que toutes les features sont présentes
        missing_features = [feat for feat in ordered_feature_names if feat not in raw_features]
        if missing_features:
            logger.error(f"Features manquantes: {missing_features}")
            raise ValueError(f"Features manquantes: {missing_features}")
        
        logger.info("Extraction des features terminée avec succès")
        logger.debug(f"Features extraites: {features_ordered}")
        
        # Retourne le dictionnaire des features dans l'ordre correct
        return features_ordered

    def _calculate_entropy(self, data):
        """Calcule l'entropie d'une liste de données"""
        if not data:
            return 0
        
        counts = Counter(data)
        probabilities = [count/len(data) for count in counts.values()]
        return -sum(p * np.log2(p) for p in probabilities)

    def analyze_wallet(self, wallet_address: str, max_transactions: int = 100) -> Dict:
        """Analyse un wallet avec détection précoce des bots après 80 transactions"""
        logger.info(f"Début de l'analyse du wallet: {wallet_address}")
        logger.info(f"Nombre maximum de transactions à analyser: {max_transactions}")
        
        all_transactions = []
        swaps_data = []
        start_time = time.time()
        BOT_THRESHOLD = 0.75  # Seuil de probabilité pour classifier comme bot
        EARLY_DETECTION_COUNT = 80  # Nombre de transactions pour la détection précoce
        
        # Récupération des signatures
        logger.info("Récupération des signatures...")
        sig_response = self._make_rpc_request(
            "getSignaturesForAddress",
            [wallet_address, {"limit": max_transactions}]
        )
        
        if "error" in sig_response:
            logger.error(f"Erreur lors de la récupération des signatures: {sig_response['error']}")
            return {"error": "Erreur lors de la récupération des signatures"}
        
        signatures = sig_response.get('result', [])
        total_signatures = len(signatures)
        logger.info(f"Nombre de signatures récupérées: {total_signatures}")
        
        # Analyse des transactions
        early_detection_triggered = False
        for idx, sig_info in enumerate(signatures):
            logger.info(f"Analyse de la transaction {idx + 1}/{total_signatures}")
            
            tx_response = self._make_rpc_request(
                "getTransaction",
                [sig_info['signature'], {"encoding": "jsonParsed", "maxSupportedTransactionVersion": 0}]
            )
            
            if "error" in tx_response:
                logger.error(f"Erreur sur la transaction {sig_info['signature']}: {tx_response['error']}")
                continue
                
            transaction = tx_response.get('result')
            if not transaction:
                logger.warning(f"Transaction vide: {sig_info['signature']}")
                continue
                
            all_transactions.append(transaction)
            
            # Vérification si c'est un swap
            is_swap, protocol = self._is_swap_transaction(transaction)
            
            if is_swap:
                tokens_in, tokens_out = self._analyze_token_changes(transaction)
                if tokens_in or tokens_out:
                    swap_info = {
                        'signature': sig_info['signature'],
                        'timestamp': sig_info.get('blockTime'),
                        'protocol': protocol,
                        'tokens_in': [{'amount': amount, 'symbol': symbol} for amount, symbol in tokens_in],
                        'tokens_out': [{'amount': amount, 'symbol': symbol} for amount, symbol in tokens_out]
                    }
                    swaps_data.append(swap_info)
                    logger.info(f"Swap détecté: {swap_info}")
            
            # Détection précoce des bots après EARLY_DETECTION_COUNT transactions
            if len(all_transactions) == EARLY_DETECTION_COUNT and self.model:
                logger.info(f"Analyse précoce après {EARLY_DETECTION_COUNT} transactions...")
                features = self._extract_features(all_transactions)
                try:
                    features_df = pd.DataFrame([features])
                    early_bot_probability = self.model.predict_proba(features_df)[0][1]
                    logger.info(f"Probabilité bot précoce: {early_bot_probability:.2%}")
                    
                    if early_bot_probability >= BOT_THRESHOLD:
                        early_detection_triggered = True
                        logger.warning(f"⚠️ BOT DÉTECTÉ avec une probabilité de {early_bot_probability:.2%}")
                        logger.warning("Arrêt anticipé de l'analyse")
                        break
                    else:
                        logger.info("Comportement humain détecté, poursuite de l'analyse...")
                        
                except Exception as e:
                    logger.error(f"Erreur lors de la classification précoce: {e}")
            
            # Pause pour respecter le rate limiting
            time.sleep(0.1)
        
        # Classification finale
        logger.info("Classification finale du wallet...")
        features = self._extract_features(all_transactions)
        final_bot_probability = 0
        
        if self.model:
            try:
                features_df = pd.DataFrame([features])
                final_bot_probability = self.model.predict_proba(features_df)[0][1]
                logger.info(f"Probabilité bot finale: {final_bot_probability:.2%}")
            except Exception as e:
                logger.error(f"Erreur lors de la classification finale: {e}")
        
        execution_time = time.time() - start_time
        results = {
            'wallet_address': wallet_address,
            'total_transactions': len(all_transactions),
            'bot_probability': float(final_bot_probability),
            'early_detection': {
                'triggered': early_detection_triggered,
                'transactions_analyzed': len(all_transactions)
            },
            'swaps': swaps_data,
            'execution_time': execution_time
        }
        
        # Messages de conclusion
        if early_detection_triggered:
            logger.warning(f"""
            ⚠️ ALERTE BOT DÉTECTÉ ⚠️
            Wallet: {wallet_address}
            Probabilité: {final_bot_probability:.2%}
            Analyse arrêtée après {len(all_transactions)} transactions
            """)
        else:
            logger.info(f"""
            ✅ Analyse complète terminée
            Wallet: {wallet_address}
            Probabilité bot: {final_bot_probability:.2%}
            Transactions analysées: {len(all_transactions)}
            """)
        
        logger.info(f"Analyse terminée en {execution_time:.2f} secondes")
        return results
def save_results(results: Dict, wallet_address: str) -> str:
    """
    Sauvegarde les résultats dans une structure de dossiers organisée.
    Retourne le chemin du fichier sauvegardé.
    """
    # Création du dossier principal pour les analyses
    base_dir = "wallet_analyses"
    
    # Sous-dossiers par date
    date_str = datetime.now().strftime("%Y-%m-%d")
    date_dir = os.path.join(base_dir, date_str)
        
    # Sous-dossier selon la classification (bot ou humain)
    is_bot = results.get('bot_probability', 0) >= 0.75
    classification_dir = "bots" if is_bot else "humans"
        
    # Chemin complet du dossier
    output_dir = os.path.join(date_dir, classification_dir)
        
    # Création des dossiers s'ils n'existent pas
    os.makedirs(output_dir, exist_ok=True)
        
    # Nom du fichier avec timestamp
    timestamp = datetime.now().strftime("%H-%M-%S")
    filename = f"{wallet_address[:8]}_{timestamp}.json"
    output_path = os.path.join(output_dir, filename)
        
    # Sauvegarde des résultats
    with open(output_path, 'w') as f:
       json.dump(results, f, indent=2)
            
    logger.info(f"Résultats sauvegardés dans: {output_path}")
    return output_path

def main():
        logger.info("Démarrage de l'analyseur de swaps Solana")
        
        analyzer = SolanaSwapAnalyzer("https://mainnet.helius-rpc.com/?api-key=0a4595b2-fcac-4086-a894-d4df21dcd82c")
        
        wallet_address = input("Entrez l'adresse du wallet à analyser: ")
        max_tx = int(input("Nombre maximum de transactions à analyser (défaut: 100): ") or "100")
        
        logger.info(f"Analyse du wallet {wallet_address}")
        results = analyzer.analyze_wallet(wallet_address, max_tx)
        
        # Affichage des résultats
        print("\n" + "="*50)
        print("RÉSULTATS DE L'ANALYSE")
        print("="*50)
        
        if results.get('early_detection', {}).get('triggered', False):
            print("\n⚠️  BOT DÉTECTÉ!")
            print(f"Analyse arrêtée après {results['early_detection']['transactions_analyzed']} transactions")
        else:
            print("\n✅ Analyse complète effectuée")
        
        print(f"\nProbabilité bot: {results['bot_probability']*100:.2f}%")
        print(f"Transactions analysées: {results['total_transactions']}")
        print(f"Swaps détectés: {len(results['swaps'])}")
        print(f"Temps d'exécution: {results['execution_time']:.2f} secondes")
        
        # Sauvegarde des résultats dans la nouvelle structure
        output_file = save_results(results, wallet_address)
        print(f"\nRésultats détaillés sauvegardés dans: {output_file}")

        # Affichage de la structure des dossiers
        base_dir = "wallet_analyses"
        if os.path.exists(base_dir):
            print("\nStructure des analyses:")
            for root, dirs, files in os.walk(base_dir):
                level = root.replace(base_dir, '').count(os.sep)
                indent = ' ' * 4 * level
                print(f"{indent}{os.path.basename(root)}/")
                subindent = ' ' * 4 * (level + 1)
                for f in files:
                    print(f"{subindent}{f}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("\nProgramme interrompu par l'utilisateur")
    except Exception as e:
        logger.exception("Erreur inattendue")
    finally:
        logger.info("Fin du programme")