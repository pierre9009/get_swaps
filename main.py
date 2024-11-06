import requests
import datetime
import time
import json
from typing import List, Dict
from base58 import b58encode, b58decode
from datetime import datetime
import joblib
import pandas as pd

class SolanaSwapAnalyzer:
    def __init__(self, rpc_url: str = "https://mainnet.helius-rpc.com/?api-key=0a4595b2-fcac-4086-a894-d4df21dcd82c"):
        print("Initialisation de SolanaSwapAnalyzer...")
        self.rpc_url = rpc_url
        self.last_request_time = 0
        self.request_count = 0
        self.RATE_LIMIT = {
            'requests_per_10s': 40,
            'min_interval': 0.25
        }
        self.call_stats = {
            "getSignaturesForAddress": 0,
            "getTransaction": 0
        }
        self.SWAP_PROGRAMS = {
          "JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4": "Jupiter",
          "whirLbMiicVdio4qvUfM5KAg6Ct8VwpYzGff3uctyCc": "Orca",
          "9W959DqEETiGZocYWCQPaJ6sBmUzgfxXfqGeTEdp3aQP": "Raydium",
          "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8": "Raydium V4",
          "SwaPpA9LAaLfeLi3a68M4DjnLqgtticKg6CnyNwgAC8": "Raydium Legacy",
          "DjVE6JNiYqPL2QXyCUUh8rNjHrbz9hXHNYt99MQ59qw1": "Orca Whirlpool",
          "CAMMCzo5YL8w4VFF8KVHrK22GGUsp5VTaW7grrKgrWqK": "Aldrin",
          "Dooar9JkhdZ7J3LHN3A7YCuoGRUggXhQaG4kijfLGU2j": "DojoSwap"
        }
        # Chargement du modèle
        print("Chargement du modèle de détection...")
        try:
          self.model = joblib.load('models/wallet_classifier_pipeline.joblib')
          print("Modèle chargé avec succès")
        except Exception as e:
          print(f"Erreur lors du chargement du modèle: {str(e)}")
          self.model = None

    def _make_rpc_request(self, method: str, params: List) -> Dict:
        """Fait une requête RPC avec gestion des rate limits"""
        print(f"\nEnvoi requête RPC: {method}")
      
        # Incrémente le compteur de calls
        if method in self.call_stats:
            self.call_stats[method] += 1
      
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
      
        if time_since_last < self.RATE_LIMIT['min_interval']:
            sleep_time = self.RATE_LIMIT['min_interval'] - time_since_last
            print(f"Rate limit: pause de {sleep_time:.2f} secondes...")
            time.sleep(sleep_time)
      
        if time_since_last > 10:
            self.request_count = 0
      
        if self.request_count >= self.RATE_LIMIT['requests_per_10s']:
            sleep_time = 10 - time_since_last
            if sleep_time > 0:
                print(f"Limite de requêtes atteinte, pause de {sleep_time:.2f} secondes...")
                time.sleep(sleep_time)
            self.request_count = 0
      
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
            self.request_count += 1
          
            print(f"Réponse reçue: Status {response.status_code}")
            return response.json()
          
        except requests.exceptions.RequestException as e:
            print(f"Erreur lors de la requête RPC: {str(e)}")
            return {"error": str(e)}

    def _analyze_token_changes(self, transaction: Dict) -> tuple:
        """Analyse les changements de tokens dans la transaction"""
        print("\nAnalyse des changements de tokens...")
      
        pretoken_balances = transaction.get('meta', {}).get('preTokenBalances', [])
        posttoken_balances = transaction.get('meta', {}).get('postTokenBalances', [])
      
        print(f"Nombre de balances pré-transaction: {len(pretoken_balances)}")
        print(f"Nombre de balances post-transaction: {len(posttoken_balances)}")
      
        pre_balances = {}
        post_balances = {}
        # Traitement des balances pré-transaction
        for balance in pretoken_balances:
            mint = balance.get('mint')
            ui_amount = balance.get('uiTokenAmount', {})
            amount = float(ui_amount.get('uiAmount', 0)) if ui_amount.get('uiAmount') is not None else 0
            decimals = ui_amount.get('decimals', 0)
            symbol = ui_amount.get('symbol', '')
            pre_balances[mint] = {
                'amount': amount,
                'decimals': decimals,
                'symbol': symbol,
                'owner': balance.get('owner', '')
            }
            print(f"Pre-balance: {symbol} = {amount} (decimals: {decimals})")
        
        # Traitement des balances post-transaction
        for balance in posttoken_balances:
            mint = balance.get('mint')
            ui_amount = balance.get('uiTokenAmount', {})
            amount = float(ui_amount.get('uiAmount', 0)) if ui_amount.get('uiAmount') is not None else 0
            decimals = ui_amount.get('decimals', 0)
            symbol = ui_amount.get('symbol', '')
            post_balances[mint] = {
                'amount': amount,
                'decimals': decimals,
                'symbol': symbol,
                'owner': balance.get('owner', '')
            }
            print(f"Post-balance: {symbol} = {amount} (decimals: {decimals})")

        tokens_in = []
        tokens_out = []
        
        all_mints = set(list(pre_balances.keys()) + list(post_balances.keys()))
        print(f"\nAnalyse des changements pour {len(all_mints)} tokens")
        
        for mint in all_mints:
            pre = pre_balances.get(mint, {'amount': 0, 'decimals': 0, 'symbol': ''})
            post = post_balances.get(mint, {'amount': 0, 'decimals': 0, 'symbol': ''})
            
            decimals = max(pre['decimals'], post['decimals'])
            symbol = post['symbol'] or pre['symbol'] or mint[:8]
            
            diff = post['amount'] - pre['amount']
            threshold = 1 / (10 ** decimals)
            
            if abs(diff) > threshold:
                if diff > 0:
                    print(f"Token IN: {diff} {symbol}")
                    tokens_in.append((diff, symbol))
                else:
                    print(f"Token OUT: {abs(diff)} {symbol}")
                    tokens_out.append((abs(diff), symbol))

        return tokens_in, tokens_out

    def _is_potential_swap(self, transaction_info: Dict) -> tuple:
        """Vérifie si une transaction pourrait être un swap basé sur ses logs"""
        if 'err' in transaction_info and transaction_info['err']:
            return False, ""

        # Récupère les logs de la transaction
        logs = transaction_info.get('logMessages', [])
        print(logs)
        if isinstance(logs, list):
            logs = '\n'.join(logs)
        else:
            logs = str(logs)

        print("\nAnalyse des logs de transaction:")
        print(logs[:200] + "...") # Affiche les 200 premiers caractères pour debug

        # Vérifie les programmes impliqués
        for program_id, protocol in self.SWAP_PROGRAMS.items():
            if program_id in logs:
                print(f"Programme de swap trouvé: {protocol}")
                return True, protocol

        # Si on arrive ici, vérifions les changements de token
        # Une transaction avec des changements de token pourrait être un swap
        if 'preTokenBalances' in str(transaction_info) and 'postTokenBalances' in str(transaction_info):
            print("Changements de token détectés")
            return True, "Unknown DEX"

        print("Aucun indicateur de swap trouvé")
        return False, ""


    def _format_time_ago(self, timestamp: int) -> str:
        now = datetime.now()
        tx_time = datetime.fromtimestamp(timestamp)
        diff = now - tx_time
        
        if diff.days > 30:
            months = diff.days // 30
            return f"{months}m ago"
        elif diff.days > 0:
            return f"{diff.days}d ago"
        elif diff.seconds > 3600:
            hours = diff.seconds // 3600
            return f"{hours}h ago"
        else:
            minutes = diff.seconds // 60
            return f"{minutes}m ago"

    

    def get_wallet_swaps(self, wallet_address: str, max_transactions: int = 100) -> List[Dict]:
      print(f"\nAnalyse du wallet: {wallet_address}")
      print(f"Nombre maximum de transactions à analyser: {max_transactions}")
      
      swaps_data = []
      collected_transactions = []
      try:
          print("\nRécupération des signatures...")
          signatures_response = self._make_rpc_request(
              "getSignaturesForAddress",
              [wallet_address, {"limit": max_transactions, "commitment": "confirmed"}]
          )
          
          if not signatures_response.get('result'):
              print("Aucune signature trouvée")
              return []
          
          signatures = signatures_response['result']
          print(f"Nombre de signatures trouvées: {len(signatures)}")
          
          for idx, sig_info in enumerate(signatures):
              print(f"\nAnalyse de la signature {idx + 1}/{len(signatures)}")
              
              tx_response = self._make_rpc_request(
                  "getTransaction",
                  [sig_info['signature'], {"encoding": "jsonParsed", "maxSupportedTransactionVersion": 0}]
              )
              
              if tx_response.get('result'):
                  tx = tx_response['result']
                  collected_transactions.append(tx)
                  
                  # Analyse des swaps comme avant
                  tokens_out, tokens_in = self._analyze_token_changes(tx)
                  if tokens_out or tokens_in:
                      # ... (code existant pour l'analyse des swaps)
                      swaps_data.append(swap_info)
              
              # Vérification tous les 10 transactions
              if len(collected_transactions) % 10 == 0:
                  is_bot, confidence = self.predict_wallet_type(collected_transactions)
                  print(f"\nAnalyse intermédiaire après {len(collected_transactions)} transactions:")
                  print(f"Probabilité bot: {confidence*100:.2f}%")
                  
                  if is_bot and confidence > 0.75:
                      print("Bot détecté avec haute confiance, arrêt de l'analyse")
                      break
              
              time.sleep(0.1)

      except Exception as e:
          print(f"Erreur lors de l'analyse: {str(e)}")
      
      # Analyse finale
      if collected_transactions:
          is_bot, confidence = self.predict_wallet_type(collected_transactions)
          print(f"\nAnalyse finale ({len(collected_transactions)} transactions):")
          print(f"Probabilité bot: {confidence*100:.2f}%")
      
      # Sauvegarde des résultats
      output_file = f"swaps_{wallet_address[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
      results = {
          'wallet_address': wallet_address,
          'transactions_analyzed': len(collected_transactions),
          'bot_probability': confidence if 'confidence' in locals() else None,
          'swaps': swaps_data
      }
      
      with open(output_file, 'w') as f:
          json.dump(results, f, indent=2)
      
      return results
    def predict_wallet_type(self, transactions):
      """Prédit si un wallet est organique ou non basé sur ses transactions"""
      if self.model is None:
          return None, 0
          
      try:
          # Extraction des features
          features = extract_features(transactions, "test_wallet")
          features_df = pd.DataFrame([features])
          
          # Prédiction
          proba = self.model.predict_proba(features_df)[0]
          is_bot = proba[0] > 0.75  # Probabilité > 75% d'être un bot
          
          return is_bot, proba[0]
      except Exception as e:
          print(f"Erreur lors de la prédiction: {str(e)}")
          return None, 0


def main():
  print("=" * 50)
  print("Démarrage de l'analyseur de swaps Solana")
  print("=" * 50)
  
  # Initialisation
  try:
      analyzer = SolanaSwapAnalyzer()
      if analyzer.model is None:
          print("\n⚠️  ATTENTION: Le modèle de détection n'a pas pu être chargé")
          proceed = input("Voulez-vous continuer sans le modèle? (o/n): ")
          if proceed.lower() != 'o':
              print("Programme terminé.")
              return
  except Exception as e:
      print(f"\n❌ Erreur lors de l'initialisation: {str(e)}")
      return

  # Input du wallet
  print("\n📝 Configuration de l'analyse")
  print("-" * 30)
  wallet_address = input("Entrez l'adresse du wallet à analyser: ")
  
  try:
      max_tx = int(input("Nombre maximum de transactions à analyser (défaut: 100): ") or "100")
  except ValueError:
      print("Valeur invalide, utilisation de la valeur par défaut: 100")
      max_tx = 100

  # Analyse
  print("\n🔍 Démarrage de l'analyse...")
  print("-" * 30)
  start_time = time.time()
  
  try:
      results = analyzer.get_wallet_swaps(wallet_address, max_tx)
      
      # Affichage des résultats
      print("\n📊 Résultats de l'analyse")
      print("-" * 30)
      print(f"Transactions analysées: {results['transactions_analyzed']}")
      
      if results.get('bot_probability') is not None:
          prob = results['bot_probability'] * 100
          print(f"Probabilité bot: {prob:.2f}%")
          if prob > 75:
              print("⚠️  ATTENTION: Ce wallet est probablement un bot!")
      
      print(f"Swaps détectés: {len(results['swaps'])}")
      
      # Statistiques des swaps
      if results['swaps']:
          protocols = {}
          tokens_traded = set()
          
          for swap in results['swaps']:
              # Comptage des protocols
              protocols[swap['protocol']] = protocols.get(swap['protocol'], 0) + 1
              
              # Collecte des tokens
              for token in swap['tokens_in']:
                  tokens_traded.add(token['symbol'])
              for token in swap['tokens_out']:
                  tokens_traded.add(token['symbol'])
          
          print("\n📈 Statistiques des swaps")
          print("-" * 30)
          print("Protocols utilisés:")
          for protocol, count in protocols.items():
              print(f"- {protocol}: {count} swaps")
          
          print(f"\nTokens tradés: {', '.join(tokens_traded)}")
      
      # Statistiques des appels API
      print("\n🌐 Statistiques des appels API")
      print("-" * 30)
      print(f"getSignaturesForAddress: {analyzer.call_stats['getSignaturesForAddress']} appels")
      print(f"getTransaction: {analyzer.call_stats['getTransaction']} appels")
      
      # Temps d'exécution
      execution_time = time.time() - start_time
      print(f"\n⏱️  Temps d'exécution: {execution_time:.2f} secondes")
      
      # Information sur le fichier de sortie
      print(f"\n💾 Les résultats ont été sauvegardés dans le fichier:")
      print(f"swaps_{wallet_address[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
      
  except Exception as e:
      print(f"\n❌ Erreur lors de l'analyse: {str(e)}")
      print("Le programme s'est terminé avec une erreur.")
      return
  
  # Option pour analyser un autre wallet
  print("\n🔄 Voulez-vous analyser un autre wallet? (o/n)")
  if input().lower() == 'o':
      main()
  else:
      print("\n✅ Programme terminé avec succès!")

if __name__ == "__main__":
  try:
      main()
  except KeyboardInterrupt:
      print("\n\n⚠️  Programme interrompu par l'utilisateur")
  except Exception as e:
      print(f"\n❌ Erreur inattendue: {str(e)}")
  finally:
      print("\n👋 Au revoir!")