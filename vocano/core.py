# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 00:14:04 2020

@author: Austin Hsu
"""

import os
# Fix OpenMP runtime conflict
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import csv
import torch
import numpy as np
import importlib
import pretty_midi
import platform
import scipy.io.wavfile as wavfile
# from apex import amp
from pathlib import Path
from tqdm import tqdm

from google_drive_downloader import GoogleDriveDownloader as gdd

from .model.PyramidNet_ShakeDrop import PyramidNet_ShakeDrop
from .model.GRU_Attention import GRU_Attention_Model, Simplified_GRU_Attention_Model
from .utils.evaluate_tools import Smooth_sdt6_modified, Naive_pitch, eval_note_acc
from .utils.dataset import EvalDataset
from .utils.est2midi import Est2MIDI

def package_check(package_name: str):
    spec = importlib.util.find_spec(package_name)
    return spec is not None

CUPY_EXIST = package_check("cupy")

from .utils.feature_extraction import test_flow as feature_extraction_np
if CUPY_EXIST:
    import cupy as cp
    from .utils.feature_extraction_cp import test_flow as feature_extraction_cp

OS_PLATFORM = platform.system()

class SingingVoiceTranscription:
    
    FILE_ID = {'PyramidNet_ShakeDrop': '1m9YT7207CXQv1KdU0ivkRQrwvPnOuR3W',
               'Patch_CNN': '1tq_LcZwWQYV7wM6dBAeDZeQn39UNkPdl'}
    DOWNLOAD_PATH = {'PyramidNet_ShakeDrop': './checkpoint/model.pt',
                     'Patch_CNN': './checkpoint/model3_patch25.npy'}
    
    def __init__(self, args):
        """
        args.keys: 
            # --- directory/filename ---
            (n)name
            (fd)feat_dir
            (pd)pitch_dir
            (md)midi_dir
            (od)output_wav_dir
            (wd)wavfile_dir
            (gd)pitch_gt_dir
            (ckpt)checkpoint_file
            # --- system ---
            (s)save_extracted
            (use_pre)use_pre_extracted
            (use_gt)use_groundtruth
            (d)device
            (use_cp)use_cp
            (use_amp)use_amp
            (bz)batch_size
            (nw)num_workers
            (pn)pin_memory
            (al)amp_level [DEPRECATED - no longer used, kept for backward compatibility]
            (model_type)model_type: 'pyramidnet' (default) or 'gru_attention' or 'simplified_gru_attention'
        """        
        self.args = args
        
        # --- backward compatibility ---
        # Set default use_amp to False if not provided
        if not hasattr(self.args, 'use_amp'):
            self.args.use_amp = False
        
        # Set default model_type to 'pyramidnet'
        if not hasattr(self.args, 'model_type'):
            self.args.model_type = 'pyramidnet'

        # --- device ---
        self._select_device()
    
    def _select_device(self):
        # cpu/selected gpu/auto-select gpu
        if self.args.device == "cpu":
            self.device = torch.device("cpu")
        elif self.args.device == "auto":
            if OS_PLATFORM == "Linux":
                os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
                memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
                for idx, memory in enumerate(memory_available):
                    print(f"cuda:{idx} available memory: {memory}")
                self.device = torch.device(f"cuda:{np.argmax(memory_available)}")
                print(f"Selected cuda:{np.argmax(memory_available)} as device")
                torch.cuda.set_device(int(np.argmax(memory_available)))
            else:
                raise OSError(f"{OS_PLATFORM} does not support auto method.")
        else:
            self.device = torch.device(f"cuda:{self.args.device}")
            torch.cuda.set_device(int(self.args.device))
    
    def _free_gpu(self):
        # free gpu usage
        if CUPY_EXIST:
            cp.get_default_memory_pool().free_all_blocks()
        torch.cuda.empty_cache()
        
    def _download_from_googledrive(self, file_id, dest_path):
        gdd.download_file_from_google_drive(file_id, dest_path)
    
    def download_ckpt(self):
        for file in self.FILE_ID:
            self._download_from_googledrive(self.FILE_ID[file], self.DOWNLOAD_PATH[file])
    
    # --- transcription --- core function
    def transcription(self):
        
        # --- download model ---
        if not Path("./checkpoint/model.pt").is_file():
            print(f"PyramidNet model checkpoint not found. Automatically download to VOCANO/checkpoint/ .")
            self._download_from_googledrive(self.FILE_ID['PyramidNet_ShakeDrop'], self.DOWNLOAD_PATH['PyramidNet_ShakeDrop'])
            print(f"PyramidNet model checkpoint downloaded.")
        if not Path("./checkpoint/model3_patch25.npy").is_file():
            print(f"Patch-CNN model checkpoint not found. Automatically download to VOCANO/checkpoint/ .")
            self._download_from_googledrive(self.FILE_ID['Patch_CNN'], self.DOWNLOAD_PATH['Patch_CNN'])
            print(f"Patch-CNN model checkpoint downloaded.")
            
        # --- feature/melody extraction ---
        if self.args.use_pre_extracted:
            feat_name = self.args.feat_dir / f"{self.args.name}_feat.npy"
            pitch_name = self.args.pitch_dir / f"{self.args.name}_pitch.npy"
            if feat_name.is_file() and pitch_name.is_file():
                self.feature = np.load(feat_name)
                self.pitch = np.load(pitch_name)
                print(f"Using pre-extracted feat {feat_name} and melody {pitch_name}")
            else:
                raise IOError(f"Given pre-extracted feature {str(feat_name)} and pitch contour {str(pitch_name)} does not exist.")
        else:
            self.data_preprocessing()

        # --- load model ---
        self.load_model()
        
        # --- vocal transcription ---
        self.voice_transcription()
        
        # --- gen midi/wav ---
        self.gen_midi()
        self.gen_wav()
        
        # --- save midi/wav ---
        self.save_midi(self.midi, self.args.midi_dir)
        self.save_wav(self.synth_midi, self.args.output_wav_dir)
        
    # --- data preprocessing ---
    def data_preprocessing(self):
        
        # --- download model ---
        if not Path("./checkpoint/model3_patch25.npy").is_file():
            print(f"Patch-CNN model checkpoint not found. Automatically download to VOCANO/checkpoint/ .")
            self._download_from_googledrive(self.FILE_ID['Patch_CNN'], self.DOWNLOAD_PATH['Patch_CNN'])
            print(f"Patch-CNN model checkpoint downloaded.")
        
        # --- feature/melody extraction ---
        self.feature, self.pitch = self.feature_extraction(self.args.wavfile_dir, self.args.use_cp)

        # --- use groundtruth pitch if selected ---
        if self.args.use_groundtruth:
            self.pitch = np.load(self.args.pitch_gt_dir)

        # --- save extracted feature/pitch ---
        if self.args.save_extracted:
            self.save_pitch_contour(self.pitch, self.args.pitch_dir)
            self.save_feature(self.feature, self.args.feat_dir)
    
    # --- feature extraction ---
    def feature_extraction(self, wavfile_dir, use_cp):
        # numpy/cupy for CFP extraction
        print(f"Feature extraction start...")
        if use_cp:
            try:
                feature, pitch = feature_extraction_cp(filename=wavfile_dir, use_ground_truth=self.args.use_groundtruth, 
                                                       batch_size=self.args.batch_size, num_workers=self.args.num_workers,
                                                       pin_memory=self.args.pin_memory, device=self.device)
            except:
                if not CUPY_EXIST:
                    raise ImportError(f"CuPy package need to be installed to enable --use_cp")
                else:
                    self._free_gpu()
                    print(f"Filesize too large. Trying with numpy solution.")
                    feature, pitch = feature_extraction_np(filename=wavfile_dir, use_ground_truth=self.args.use_groundtruth, 
                                                           batch_size=self.args.batch_size, num_workers=self.args.num_workers,
                                                           pin_memory=self.args.pin_memory, device=self.device)
        else:
            feature, pitch = feature_extraction_np(filename=wavfile_dir, use_ground_truth=self.args.use_groundtruth, 
                                                   batch_size=self.args.batch_size, num_workers=self.args.num_workers,
                                                   pin_memory=self.args.pin_memory, device=self.device)
        
        # --- free GPU ---
        self._free_gpu()
        
        print(f"Feature successfully extracted.")
        print(f"feature: {feature.shape}")
        print(f"pitch: {pitch.shape}")
        return feature, pitch
    
    # --- load model ---
    def load_model(self):
        # load melody extraction model
        # load note segmentation model
        
        # Choose model architecture based on model_type
        print(f"Loading model type: {self.args.model_type}")
        
        if self.args.model_type == 'gru_attention':
            # Use new GRU + Attention model
            self.feature_extractor = GRU_Attention_Model(
                conv1_in_channel=9, 
                num_classes=6, 
                hidden_dim=256, 
                num_layers=2
            )
        elif self.args.model_type == 'simplified_gru_attention':
            # Use simplified GRU + Attention model
            self.feature_extractor = Simplified_GRU_Attention_Model(
                conv1_in_channel=9, 
                num_classes=6, 
                hidden_dim=128
            )
        else:  # Default: pyramidnet
            # Original PyramidNet model (commented for reference)
            self.feature_extractor = PyramidNet_ShakeDrop(depth=110, alpha=270, shakedrop=True)
        
        # --- load checkpoint ---
        if self.args.model_type in ['gru_attention', 'simplified_gru_attention']:
            # For new models, check if checkpoint exists, otherwise use random initialization
            print("Note: GRU+Attention models require training. Using random initialization if checkpoint not found.")
            if Path(self.args.checkpoint_file).is_file():
                try:
                    checkpoint = torch.load(self.args.checkpoint_file, map_location=self.device, weights_only=False)
                    # Try to load compatible checkpoint
                    if 'model' in checkpoint:
                        self.feature_extractor.load_state_dict(checkpoint['model'], strict=False)
                        print("Partially loaded checkpoint weights (some layers may use random init)")
                    else:
                        print("Checkpoint format not compatible, using random initialization")
                except Exception as e:
                    print(f"Warning: Could not load checkpoint: {e}")
                    print("Using random initialization for GRU+Attention model")
            else:
                print("No checkpoint found for GRU+Attention model, using random initialization")
                print("Model needs to be trained before meaningful inference")
        else:
            # Original PyramidNet checkpoint loading
            checkpoint = torch.load(self.args.checkpoint_file, map_location=self.device, weights_only=False)
            self.feature_extractor.load_state_dict(checkpoint['model'])
        
        self.feature_extractor = self.feature_extractor.to(self.device)
        
        # --- evaluate mode ---
        self.feature_extractor.eval()
        
        # Print mode information
        precision_mode = "Mixed Precision (PyTorch AMP)" if self.args.use_amp else "Full Precision (FP32)"
        print(f"Model loaded successfully - Type: {self.args.model_type}")
        print(f"Inference mode: {precision_mode}")
        
    # --- voice transcription ---
    def voice_transcription(self):
        # --- data loader ---
        test_dataset = EvalDataset(self.feature)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.args.batch_size, shuffle=False,
                                                  num_workers=self.args.num_workers, pin_memory=self.args.pin_memory)
        
        # --- test loop ---
        # TODO: Test the actual output of the model, without softmax, and try to use DALI dataset to create training dataset in raw output format.
        outputs = []
        raw_outputs = []  # Store raw feature_extractor outputs before softmax
        
        # Create 'ond' folder for saving model outputs
        ond_dir = Path("./ond")
        ond_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine if AMP should be enabled (only on CUDA devices)
        use_amp = self.args.use_amp and self.device.type == 'cuda'
        device_type = 'cuda' if self.device.type == 'cuda' else 'cpu'
        
        for batch_idx, feat in enumerate(tqdm(test_loader)):
            feat = feat.to(self.device)
            
            # Use PyTorch native AMP for faster inference if enabled
            # Using new torch.amp.autocast API (torch.cuda.amp.autocast is deprecated)
            with torch.amp.autocast(device_type=device_type, enabled=use_amp):
                # Get raw output from feature_extractor (PyramidNet)
                raw_sdt_hat = self.feature_extractor.forward(feat)
                
                # Store raw output before softmax processing
                raw_outputs.append(raw_sdt_hat.detach().cpu())
                
                # Apply softmax processing for normal pipeline
                sdt_hat = torch.nn.functional.softmax(raw_sdt_hat.view(3,-1,2), dim=2).view(-1,6)
            
            outputs.append(sdt_hat.detach().cpu())
        
        # Concatenate all outputs
        outputs = torch.cat(outputs)
        raw_outputs = torch.cat(raw_outputs)
        
        # Save raw feature_extractor outputs to 'ond' folder
        raw_outputs_np = raw_outputs.numpy()
        raw_outputs_path = ond_dir / f"{self.args.name}_raw_outputs.npy"
        np.save(raw_outputs_path, raw_outputs_np)
        print(f"Raw feature_extractor outputs saved to {raw_outputs_path}")
        
        # Also save as CSV for easier inspection (first 1000 samples)
        csv_path = ond_dir / f"{self.args.name}_raw_outputs.csv"
        sample_size = min(1000, raw_outputs_np.shape[0])
        with open(csv_path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['sample_idx'] + [f'output_{i}' for i in range(raw_outputs_np.shape[1])])
            for i in range(sample_size):
                csv_writer.writerow([i] + raw_outputs_np[i].tolist())
        print(f"Raw feature_extractor outputs sample (first {sample_size} samples) saved to {csv_path}")
        
        # --- post processing ---
        p_np = self.pitch
        predict_on_notes_np = outputs.numpy()
        # print(f"predict_on_notes_np: {predict_on_notes_np.shape}")

        pitch_intervals, sSeq_np, dSeq_np, onSeq_np, offSeq_np, conflict_ratio = Smooth_sdt6_modified(predict_on_notes_np, threshold=0.5) # list of onset secs, ndarray
        print(f"pitch_intervals: {pitch_intervals.shape}")
        # print(f"sSeq_np: {sSeq_np.shape}")
        # print(f"dSeq_np: {dSeq_np.shape}")
        # print(f"onSeq_np: {onSeq_np.shape}")
        # print(f"offSeq_np: {offSeq_np.shape}")
        print(f"conflict_ratio: {conflict_ratio}")

        freq_est = Naive_pitch(p_np, pitch_intervals)
            
        # --- for midi ---
        self.est = np.hstack((pitch_intervals, freq_est.reshape((-1,1))))
        print(f"self.est:{self.est}")

        self.sdt = np.hstack((sSeq_np.reshape((-1,1)), dSeq_np.reshape((-1,1)), onSeq_np.reshape((-1,1)), offSeq_np.reshape((-1,1))))

        # --- for evaluation with note accuracy ---
        # Load ground-truth note sequence (frequencies) if available and compute sequence-level accuracy
        try:
            gt_notes_path = Path("D:/FYP/AI-Vocal-Transcription/VOCANO/dataset/gt_notes") / f"{self.args.name}.csv"
            if gt_notes_path.is_file():
                gt_csv = np.genfromtxt(gt_notes_path, delimiter=",", skip_header=1)
                # Column order: start_time, end_time, frequency
                self.gt_notes = gt_csv[:, 2].astype(float)
                est_notes = self.est[:, 2].astype(float)
                note_accuracy = eval_note_acc(self.gt_notes, est_notes)
                print(f"note_accuracy: {note_accuracy}")
            else:
                print(self.args.name)
                print(f"Ground-truth note file not found: {gt_notes_path}. Skip note accuracy.")
        except Exception as e:
            print(f"Failed to compute note accuracy: {e}")

        # experiment for NAcc: 
        # song: A_Day_To_Remember-Casablanca_Sucked_Anyways
        # note_accuracy: 0.4864864864864865

        # --- free GPU ---
        self._free_gpu()
        
    # --- gen midi ---
    def gen_midi(self):
        # generate midi file from prediction or from groundtruth
        self.midi = Est2MIDI(self.est)
        
    # --- gen wav ---
    def gen_wav(self):
        # generate wav from midi file
        self.synth_midi = self.midi.synthesize().astype(np.float32)
        
    # --- save midi ---
    def save_midi(self, midi: pretty_midi.pretty_midi.PrettyMIDI, save_dir: Path):
        # save midi file
        print(f"Writing midi...")
        save_dir.mkdir(parents=True, exist_ok=True)
        midi.write(str(save_dir / f"{self.args.name}.mid"))
        print(f"Midi successfully saved to {save_dir}")        
        
    # --- save wav ---
    def save_wav(self, synth_midi: np.ndarray, save_dir: Path):
        # save wav file
        print(f"Writing wav...")
        save_dir.mkdir(parents=True, exist_ok=True)
        wavfile.write(save_dir / f"{self.args.name}.wav", 44100, synth_midi)
        print(f"Wav successfully saved to {save_dir}")        
        
    # --- save pitch contour ---
    def save_pitch_contour(self, pitch: np.ndarray, save_dir: Path):
        # save extracted melody
        print(f"Writing pitch contour information...")
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as .npy file
        np.save(save_dir / f"{self.args.name}_pitch.npy", pitch)
        
        # Save as CSV file with format (frame_index, frame_time, pitch)
        # Frame rate: fs=16000Hz, Hop=320 samples -> frame_time = 0.02s per frame
        frame_time_per_frame = 320.0 / 16000.0  # 0.02 seconds
        num_frames = len(pitch)
        
        csv_data = []
        for frame_idx in range(num_frames):
            frame_time = frame_idx * frame_time_per_frame
            csv_data.append([frame_idx, frame_time, pitch[frame_idx]])
        
        # Write to CSV
        csv_path = save_dir / f"{self.args.name}_pitch.csv"
        with open(csv_path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['frame_index', 'frame_time', 'pitch'])  # Header
            csv_writer.writerows(csv_data)
        
        print(f"Pitch contour successfully saved to {save_dir} (.npy and .csv)")
        
    # --- save feature ---
    def save_feature(self, feature: np.ndarray, save_dir: Path):
        # save extracted feature
        print(f"Writing CFP feature information...")
        save_dir.mkdir(parents=True, exist_ok=True)
        np.save(save_dir / f"{self.args.name}_feat.npy", feature)
        print(f"Feature successfully saved to {save_dir}")        