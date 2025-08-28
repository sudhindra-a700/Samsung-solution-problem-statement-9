#!/usr/bin/env python3
"""
Fast Massive Dataset Generator
Optimized for speed and large data generation
"""

import pandas as pd
import numpy as np
import time
import os
from datetime import datetime
import gc

class FastMassiveGenerator:
    """Fast generator for large Reel vs Non-Reel datasets"""
    
    def __init__(self):
        self.output_dir = "./fast_massive_dataset"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Streamlined configurations
        self.apps = ['youtube', 'instagram', 'tiktok', 'facebook', 'twitter', 'snapchat']
        self.reel_prob = {'youtube': 0.4, 'instagram': 0.7, 'tiktok': 0.95, 'facebook': 0.6, 'twitter': 0.5, 'snapchat': 0.9}
        
        # Pre-computed arrays for speed
        self.reel_formats = np.array([144, 240, 360, 480, 720])
        self.nonreel_formats = np.array([720, 1080, 1440, 2160])
        self.fps_options = np.array([24, 25, 30, 60])
        self.audio_formats = np.array([64, 128, 192, 256])
        
        # Server IPs
        self.servers = {
            'youtube': ['172.217.22.106', '74.125.13.143'],
            'instagram': ['31.13.82.52', '157.240.22.35'],
            'tiktok': ['104.18.6.188', '104.18.7.188'],
            'facebook': ['31.13.82.52', '157.240.22.35'],
            'twitter': ['104.244.42.193', '104.244.42.129'],
            'snapchat': ['35.186.224.25', '35.186.224.47']
        }
        
    def generate_chunk(self, chunk_size: int, chunk_id: int) -> pd.DataFrame:
        """Generate a chunk of data efficiently"""
        
        np.random.seed(chunk_id)  # Reproducible per chunk
        
        # Pre-allocate arrays
        records = []
        
        for i in range(chunk_size):
            session_id = chunk_id * chunk_size + i
            
            # Fast random selections
            app = np.random.choice(self.apps)
            is_reel = np.random.random() < self.reel_prob[app]
            
            # Video characteristics
            if is_reel:
                fmt = np.random.choice(self.reel_formats)
                duration = np.random.randint(15, 180)
                num_records = np.random.randint(3, 8)  # 3-7 records per session
            else:
                fmt = np.random.choice(self.nonreel_formats)
                duration = np.random.randint(300, 3600)
                num_records = np.random.randint(5, 15)  # 5-14 records per session
            
            fps = np.random.choice(self.fps_options)
            audio_fmt = np.random.choice(self.audio_formats)
            server_ip = np.random.choice(self.servers[app])
            
            # Generate multiple records per session
            base_time = time.time() - np.random.randint(0, 604800)
            buffer_health = np.random.randint(5000, 15000)
            played_frames = 0
            dropped_frames = 0
            
            for record_num in range(num_records):
                current_time = base_time + (record_num * duration / num_records)
                
                # Update session state
                frames_added = fps * (duration / num_records)
                played_frames += int(frames_added)
                
                # Buffer simulation
                if is_reel:
                    buffer_change = np.random.randint(-300, 800)
                    buffer_health = np.clip(buffer_health + buffer_change, 1000, 12000)
                else:
                    buffer_change = np.random.randint(-800, 1200)
                    buffer_health = np.clip(buffer_health + buffer_change, 2000, 25000)
                
                # Stalling
                stalling = 1 if buffer_health < 2000 else 0
                if stalling:
                    dropped_frames += np.random.randint(1, 4)
                
                # Quality changes
                qc = 1 if np.random.random() < 0.15 else 0
                qc_to = fmt
                if qc:
                    if is_reel:
                        qc_to = np.random.choice([360, 480, 720])
                    else:
                        qc_to = np.random.choice([720, 1080])
                
                # Phase
                if stalling:
                    phase = "stalling"
                elif buffer_health > 18000:
                    phase = "filling"
                else:
                    phase = "depletion"
                
                # Create record
                timestamp_str = datetime.fromtimestamp(current_time).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                
                record = {
                    'timestamp': timestamp_str,
                    'fmt': qc_to if qc else fmt,
                    'fps': fps,
                    'afmt': audio_fmt,
                    'bh': buffer_health,
                    'droppedFrames': dropped_frames,
                    'playedFrames': played_frames,
                    'videoid': f"{app}_{session_id:08d}",
                    'stalling': stalling,
                    'qc': qc,
                    'qcTo': qc_to,
                    'phase': phase,
                    'app': app,
                    'traffic_type': 'REEL' if is_reel else 'NON_REEL',
                    'label': 1 if is_reel else 0
                }
                
                records.append(record)
        
        return pd.DataFrame(records)
    
    def generate_network_chunk(self, chunk_size: int, chunk_id: int) -> pd.DataFrame:
        """Generate network traffic chunk"""
        
        np.random.seed(chunk_id + 10000)  # Different seed for network data
        
        packets = []
        
        for i in range(chunk_size * 50):  # 50 packets per session on average
            session_id = chunk_id * chunk_size + (i // 50)
            
            app = np.random.choice(self.apps)
            is_reel = np.random.random() < self.reel_prob[app]
            server_ip = np.random.choice(self.servers[app])
            
            # Packet characteristics
            packet_time = time.time() - np.random.randint(0, 604800) + (i * 0.01)
            
            # 70% download, 30% upload
            if np.random.random() < 0.7:
                src_ip = server_ip
                dst_ip = "10.10.0.140"
                src_port = 443
                dst_port = np.random.randint(40000, 65000)
                
                if is_reel:
                    packet_size = np.random.choice([600, 800, 1000, 1200])
                else:
                    packet_size = np.random.choice([1000, 1200, 1400, 1500])
            else:
                src_ip = "10.10.0.140"
                dst_ip = server_ip
                src_port = np.random.randint(40000, 65000)
                dst_port = 443
                packet_size = np.random.choice([50, 100, 150])
            
            protocol = 17 if np.random.random() < 0.8 else 6
            
            packet = {
                'timestamp': packet_time,
                'ipSrc': src_ip,
                'ipDst': dst_ip,
                'tcpPortSrc': src_port if protocol == 6 else '',
                'tcpPortDst': dst_port if protocol == 6 else '',
                'udpPortSrc': src_port if protocol == 17 else '',
                'udpPortDst': dst_port if protocol == 17 else '',
                'tcpLen': packet_size if protocol == 6 else '',
                'udpLen': packet_size if protocol == 17 else '',
                'payloadProtocolNumber': protocol
            }
            
            packets.append(packet)
        
        return pd.DataFrame(packets)
    
    def generate_massive_dataset(self, target_mb: int = 1000):
        """Generate massive dataset with target size in MB"""
        
        print(f"🚀 Fast Massive Dataset Generation")
        print("=" * 60)
        print(f"🎯 Target Size: {target_mb} MB")
        
        chunk_size = 1000  # Sessions per chunk
        estimated_mb_per_chunk = 5  # Estimated MB per chunk
        total_chunks = max(1, target_mb // estimated_mb_per_chunk)
        
        print(f"📊 Chunk Size: {chunk_size:,} sessions")
        print(f"🔢 Total Chunks: {total_chunks:,}")
        
        start_time = time.time()
        total_size = 0
        
        # Generate application data
        print(f"\n📱 Generating application data...")
        app_files = []
        
        for chunk_id in range(total_chunks):
            df_chunk = self.generate_chunk(chunk_size, chunk_id)
            
            # Save chunk
            chunk_file = os.path.join(self.output_dir, f"app_chunk_{chunk_id:04d}.csv")
            df_chunk.to_csv(chunk_file, index=False)
            app_files.append(chunk_file)
            
            chunk_size_mb = os.path.getsize(chunk_file) / 1024 / 1024
            total_size += os.path.getsize(chunk_file)
            
            if (chunk_id + 1) % 10 == 0:
                progress = (chunk_id + 1) / total_chunks * 100
                current_mb = total_size / 1024 / 1024
                print(f"   ✅ Chunk {chunk_id + 1:,}/{total_chunks:,} ({progress:.1f}%) - {current_mb:.1f} MB")
            
            # Clean up memory
            del df_chunk
            gc.collect()
            
            # Check if target reached
            if total_size >= target_mb * 1024 * 1024:
                print(f"🎯 Target size reached!")
                break
        
        # Generate network data
        print(f"\n🌐 Generating network traffic...")
        network_files = []
        
        network_chunks = min(total_chunks // 2, 50)  # Fewer network chunks
        
        for chunk_id in range(network_chunks):
            df_network = self.generate_network_chunk(chunk_size, chunk_id)
            
            # Save chunk
            chunk_file = os.path.join(self.output_dir, f"network_chunk_{chunk_id:04d}.csv")
            df_network.to_csv(chunk_file, index=False)
            network_files.append(chunk_file)
            
            total_size += os.path.getsize(chunk_file)
            
            if (chunk_id + 1) % 5 == 0:
                progress = (chunk_id + 1) / network_chunks * 100
                print(f"   ✅ Network chunk {chunk_id + 1:,}/{network_chunks:,} ({progress:.1f}%)")
            
            del df_network
            gc.collect()
        
        # Combine files
        print(f"\n🔗 Combining files...")
        
        # Combine application files
        print(f"   📱 Combining {len(app_files)} application files...")
        combined_app = os.path.join(self.output_dir, "massive_application_data.csv")
        
        with open(combined_app, 'w') as outfile:
            for i, app_file in enumerate(app_files):
                with open(app_file, 'r') as infile:
                    if i == 0:
                        outfile.write(infile.read())  # Include header
                    else:
                        next(infile)  # Skip header
                        outfile.write(infile.read())
                os.remove(app_file)  # Clean up
        
        # Combine network files
        print(f"   🌐 Combining {len(network_files)} network files...")
        combined_network = os.path.join(self.output_dir, "massive_network_traffic.csv")
        
        with open(combined_network, 'w') as outfile:
            for i, network_file in enumerate(network_files):
                with open(network_file, 'r') as infile:
                    if i == 0:
                        outfile.write(infile.read())  # Include header
                    else:
                        next(infile)  # Skip header
                        outfile.write(infile.read())
                os.remove(network_file)  # Clean up
        
        # Create balanced training set
        print(f"   ⚖️  Creating balanced training dataset...")
        
        # Read application data and create balanced set
        df_app = pd.read_csv(combined_app)
        
        reel_data = df_app[df_app['label'] == 1]
        nonreel_data = df_app[df_app['label'] == 0]
        
        # Balance (50/50 split)
        min_count = min(len(reel_data), len(nonreel_data), 50000)  # Max 50k each
        
        balanced_df = pd.concat([
            reel_data.sample(n=min_count, random_state=42),
            nonreel_data.sample(n=min_count, random_state=42)
        ]).sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Save balanced training set
        training_file = os.path.join(self.output_dir, "massive_balanced_training.csv")
        balanced_df.to_csv(training_file, index=False)
        
        # Generate stats
        app_size = os.path.getsize(combined_app)
        network_size = os.path.getsize(combined_network)
        training_size = os.path.getsize(training_file)
        total_final_size = app_size + network_size + training_size
        
        elapsed = time.time() - start_time
        
        print(f"\n🎉 Fast Massive Dataset Complete!")
        print("=" * 60)
        print(f"📊 Application Data: {len(df_app):,} records ({app_size/1024/1024:.1f} MB)")
        print(f"🌐 Network Traffic: {len(pd.read_csv(combined_network)):,} packets ({network_size/1024/1024:.1f} MB)")
        print(f"⚖️  Balanced Training: {len(balanced_df):,} records ({training_size/1024/1024:.1f} MB)")
        print(f"📁 Total Size: {total_final_size/1024/1024:.1f} MB")
        print(f"⏱️  Generation Time: {elapsed:.1f} seconds")
        print(f"🚀 Speed: {total_final_size/1024/1024/elapsed:.1f} MB/second")
        
        # Distribution stats
        reel_count = len(balanced_df[balanced_df['label'] == 1])
        nonreel_count = len(balanced_df[balanced_df['label'] == 0])
        
        print(f"\n🏷️  Training Set Distribution:")
        print(f"   REEL: {reel_count:,} ({reel_count/len(balanced_df)*100:.1f}%)")
        print(f"   NON-REEL: {nonreel_count:,} ({nonreel_count/len(balanced_df)*100:.1f}%)")
        
        app_dist = df_app['app'].value_counts()
        print(f"\n📱 App Distribution:")
        for app, count in app_dist.items():
            print(f"   {app.upper()}: {count:,} ({count/len(df_app)*100:.1f}%)")
        
        print(f"\n✅ Dataset ready for SAC-GRU training!")
        print(f"📁 Location: {self.output_dir}")
        
        return total_final_size

def main():
    """Main function"""
    
    print("🎯 Fast Massive Dataset Generator")
    print("Choose target size:")
    print("1. 500 MB (Fast)")
    print("2. 1 GB (Medium)")
    print("3. 2 GB (Large)")
    print("4. 5 GB (Massive)")
    
    choice = input("Enter choice (1-4) or custom MB: ").strip()
    
    if choice == '1':
        target_mb = 500
    elif choice == '2':
        target_mb = 1000
    elif choice == '3':
        target_mb = 2000
    elif choice == '4':
        target_mb = 5000
    else:
        try:
            target_mb = int(choice)
        except:
            target_mb = 1000
    
    generator = FastMassiveGenerator()
    generator.generate_massive_dataset(target_mb)

if __name__ == "__main__":
    main()

