import pandas as pd

def calculate_improvement():
    print("📋 ANALISIS TARGET KINERJA")
    print("-" * 40)
    
    # Load data
    df_rl = pd.read_csv('logs/training_log_FINAL_MAPPO_LSTM.csv')
    df_base = pd.read_csv('logs/baseline_results.csv')
    
    rl = df_rl.iloc[-1]
    fifo = df_base[df_base['Mode'] == 'fifo'].iloc[0]

    # 1. Analisis Throughput (Target: Min +15%)
    tp_imp = ((rl['total_throughput_tons'] - fifo['Throughput_Tons']) / fifo['Throughput_Tons']) * 100
    print(f"Throughput Improvement : {tp_imp:.2f}% (Target: >15%)")
    print(f"STATUS: {'✅ LOLOS' if tp_imp >= 15 else '❌ GAGAL'}")

    # 2. Analisis Delay per Ton (Target: Min -35%)
    # Menggunakan delay normalisasi (delay total / throughput) agar adil
    rl_delay_per_ton = rl['total_delay_minutes'] / rl['total_throughput_tons']
    fifo_delay_per_ton = fifo['Delay_Minutes'] / fifo['Throughput_Tons']
    
    delay_imp = ((fifo_delay_per_ton - rl_delay_per_ton) / fifo_delay_per_ton) * 100
    print(f"Delay Reduction per Ton: {delay_imp:.2f}% (Target: >35%)")
    print(f"STATUS: {'✅ LOLOS' if delay_imp >= 35 else '❌ GAGAL'}")

    # 3. Analisis Antrean (Target: < 45 Menit)
    print(f"Average Queue Time     : {rl['avg_queue_time']:.2f} menit (Target: <45)")
    print(f"STATUS: {'✅ LOLOS' if rl['avg_queue_time'] < 45 else '❌ GAGAL'}")
    print("-" * 40)

if __name__ == "__main__":
    calculate_improvement()