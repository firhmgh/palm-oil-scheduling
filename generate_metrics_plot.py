import matplotlib.pyplot as plt
import numpy as np

methods = ['FIFO (Baseline)', 'Oldest First', 'Model Optimized (MAPPO LSTM)']
queue_times = [0.05, 0.06, 0.62]  # Rata-rata avg_queue_time 
throughputs = [96.00, 115.20, 7638.55]  # Rata-rata throughput_tons (ton)
rewards = [648.00, -40.00, 636.47]  # Rata-rata reward

fig, ax1 = plt.subplots(figsize=(10, 6))

# Bar untuk queue time (kiri)
bars = ax1.bar(methods, queue_times, color=['#ef4444', '#f59e0b', '#10b981'], alpha=0.7, label='Rata-rata Durasi Antrian (unit waktu)')
ax1.set_ylabel('Rata-rata Durasi Antrian', color='#ef4444')
ax1.tick_params(axis='y', labelcolor='#ef4444')

# Garis untuk throughput (kanan)
ax2 = ax1.twinx()
ax2.plot(methods, throughputs, 'o-', color='#3b82f6', linewidth=2, markersize=8, label='Rata-rata Throughput (ton)')
ax2.set_ylabel('Rata-rata Throughput (ton)', color='#3b82f6')
ax2.tick_params(axis='y', labelcolor='#3b82f6')

# Judul & label
plt.title('Metrik Performa: Sebelum vs Sesudah Optimasi Penjadwalan Kelapa Sawit', fontsize=14, pad=20)
ax1.set_xlabel('Metode Penjadwalan')
fig.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2)

# Tambah anotasi untuk queue times
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, height + 0.01, f'{height:.2f}', ha='center', va='bottom')

# Simpan gambar 
plt.tight_layout()
plt.savefig('assets/images/performance-metrics.png', dpi=300, bbox_inches='tight')
plt.show()  # Preview plot