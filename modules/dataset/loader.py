import os
import random
import torch
import numpy as np

import csv

import librosa

from tqdm import tqdm

from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset


def get_datasets(data_csv: str, speaker_file: str):
    if not os.path.isfile(data_csv):
        raise FileNotFoundError(f'metadata csv not found: {data_csv}')

    with open(data_csv, encoding='utf-8') as f:
        reader = csv.reader(f)
        lines = [row for row in reader]
        headers = lines[0]
        datas = {}
        for line in lines[1:]:
            datas[line[0]] = {}
            for h, v in enumerate(line[1:], start=1):
                datas[line[0]][headers[h]] = v

    if os.path.isfile(speaker_file):
        with open(speaker_file, encoding='utf-8') as f:
            reader = csv.reader(f)
            spk_infos = {row[0]: row[1] for row in reader}
    else:
        spk_infos = None

    return datas, spk_infos


class AudioDataset(TorchDataset):
    def __init__(
        self,
        root_path,
        metadatas: dict,
        spk_infos: dict,
        crop_duration,
        hop_size,
        sampling_rate,
        whole_audio=False,
        device='cpu',
        fp16=False,
        use_aug=False,
        use_spk_embed=False,
        units_only=False,
    ):
        super().__init__()

        self.root_path = root_path
        self.crop_duration = crop_duration
        self.sampling_rate = sampling_rate
        self.hop_size = hop_size
        self.whole_audio = whole_audio
        self.use_aug = use_aug
        self.use_spk_embed = use_spk_embed

        self.units_only = units_only

        self.paths = list(metadatas.keys())

        self.data_buffer = {}
        self.spk_embeds = {}

        skip_index = []
        for idx, file in enumerate(tqdm(self.paths, total=len(self.paths), desc='loading data')):
            audio_path = os.path.join(root_path, file)
            duration = librosa.get_duration(path=audio_path, sr=sampling_rate)

            if duration < crop_duration + 0.1 and not whole_audio:
                print(f"skip loading file {file}, because length {duration:.2f}sec is too short.")
                skip_index.append(idx)
                continue

            file_dir, file_name = os.path.split(file)
            file_rel = os.path.relpath(file_dir, start='data')

            # load f0
            f0_path = os.path.join(self.root_path, 'f0', file_rel, file_name) + '.npz'
            f0 = np.load(f0_path)['f0']
            f0 = torch.from_numpy(f0).float().unsqueeze(-1).to(device)

            # load audio
            audio, sr = librosa.load(audio_path, sr=sampling_rate)
            audio = torch.from_numpy(audio).to(device)

            # load volume
            volume_path = os.path.join(self.root_path, 'volume', file_rel, file_name) + '.npz'
            volume = np.load(volume_path)['volume']
            volume = torch.from_numpy(volume).float().unsqueeze(-1).to(device)

            # load units
            units_dir = os.path.join(self.root_path, 'units', file_rel, file_name) + '.npz'
            units = np.load(units_dir)['units']
            units = torch.from_numpy(units).to(device)

            if fp16:
                audio = audio.half()
                units = units.half()

            self.data_buffer[file] = {
                'duration': duration,
                'audio': audio,
                'units': units,
                'f0': f0,
                'volume': volume,
                'spk_id': torch.LongTensor(np.array([int(metadatas[file]['spk_id'])])).to(device),
            }

            if use_spk_embed and spk_infos is not None:
                spk_id = metadatas[file]['spk_id']
                if spk_id in spk_infos:
                    self.spk_embeds[spk_id] = torch.from_numpy(np.array(spk_infos[spk_id], dtype=np.float32)).to(device)
                else:
                    print(f"Warning: spk_id {spk_id} not found in spk_info.csv")
                    self.data_buffer[file]['spk_embed'] = torch.zeros(1, dtype=torch.float32, device=device)
            elif use_spk_embed:
                self.data_buffer[file]['spk_embed'] = torch.zeros(1, dtype=torch.float32, device=device)

        if len(skip_index) > 0:
            print(f"skip {len(skip_index)} files.")
            self.paths = [v for i, v in enumerate(self.paths) if i not in skip_index]

    def __getitem__(self, file_idx):
        file = self.paths[file_idx]
        data_buffer = self.data_buffer[file]

        # # check duration. if too short, then skip
        if data_buffer['duration'] < (self.crop_duration + 0.1):
            return self.__getitem__((file_idx + 1) % len(self.paths))

        # get item
        return self.get_data(file, data_buffer)

    def __len__(self):
        return len(self.paths)

    def get_data(self, file, data_buffer):
        name = os.path.splitext(file)[0]
        frame_resolution = self.hop_size / self.sampling_rate
        duration = data_buffer['duration']
        crop_duration = duration if self.whole_audio else self.crop_duration

        idx_from = 0 if self.whole_audio else random.uniform(0, duration - crop_duration - 0.1)
        start_frame = int(idx_from / frame_resolution)
        units_frame_len = int(crop_duration / frame_resolution)

        # load units
        units = data_buffer['units'][start_frame : start_frame + units_frame_len]
        # load spk_id
        spk_id = data_buffer['spk_id']

        if self.units_only:
            return dict(spk_id=spk_id, units=units)

        # load audio
        audio = data_buffer['audio'][start_frame*self.hop_size:(start_frame + units_frame_len)*self.hop_size]

        # load f0
        f0 = data_buffer['f0'][start_frame : start_frame + units_frame_len]

        # load volume
        volume = data_buffer['volume'][start_frame : start_frame + units_frame_len]

        # volume augumentation
        if self.use_aug:
            max_gain = torch.max(torch.abs(audio)) + 1e-5
            max_shift = min(1.5, torch.log10(1./max_gain))
            log10_vol_shift = random.uniform(-1.5, max_shift)
            aug_audio = audio*(10 ** log10_vol_shift)
            aug_volume = volume*(10 ** log10_vol_shift)
        else:
            aug_audio = audio
            aug_volume = volume

        data = {
            'audio': aug_audio,
            'f0': f0,
            'volume': aug_volume,
            'units': units,
            'spk_id': spk_id,
            'name': name
        }

        if self.use_spk_embed:
            data['spk_embed'] = self.data_buffer[file].get('spk_embed', None)

        return data



class AudioCrop:
    def __init__(self, block_size, sampling_rate, crop_duration):
        self.block_size = block_size
        self.sampling_rate = sampling_rate
        self.crop_duration = crop_duration

    def crop_audio(self, batch):
        frame_resolution = self.block_size / self.sampling_rate
        units_frame_len = int(self.crop_duration / frame_resolution)
        for b in range(len(batch['audio'])):
            duration = len(batch['audio'][b]) / self.sampling_rate
            idx_from = random.uniform(0, duration - self.crop_duration - 0.1)
            start_frame = int(idx_from / frame_resolution)

            batch['units'][b] = batch['units'][b][start_frame:start_frame+units_frame_len]
            batch['f0'][b] = batch['f0'][b][start_frame:start_frame+units_frame_len]
            batch['volume'][b] = batch['volume'][b][start_frame:start_frame+units_frame_len]

            batch['audio'][b] = batch['audio'][b][start_frame*self.block_size:(start_frame + units_frame_len)*self.block_size]

        for b in range(len(batch['audio'])):
            batch['units'] = torch.tensor(batch['units'])
            batch['f0'] = torch.tensor(batch['f0'])
            batch['volume'] = torch.tensor(batch['volume'])
            batch['audio'] = torch.tensor(batch['audio'])
            if 'spk_embed' in batch:
                batch['spk_embed'] = torch.tensor(batch['spk_embed'])
            batch['spk_id'] = torch.tensor(batch['spk_id'])

        return batch


def get_data_loaders(args):
    loaders = {}

    ds_train, spk_infos = get_datasets(os.path.join(args.data.dataset_path, 'train.csv'), os.path.join(args.data.dataset_path, 'spk_info.csv'))

    loaders['train'] = DataLoader(
        AudioDataset(
            root_path=args.data.dataset_path,
            metadatas=ds_train,
            spk_infos=spk_infos,
            crop_duration=args.data.duration,
            hop_size=args.data.block_size,
            sampling_rate=args.data.sampling_rate,
            whole_audio=False,
            device=args.train.cache_device,
            fp16=args.train.cache_fp16,
            use_aug=True,
            use_spk_embed=args.model.use_speaker_embed,
            units_only=args.train.only_u2c_stack),
        batch_size=args.train.batch_size,
        shuffle=True,
        num_workers=args.train.num_workers if args.train.cache_device=='cpu' else 0,
        persistent_workers=(args.train.num_workers > 0) if args.train.cache_device=='cpu' else False,
        pin_memory=True if args.train.cache_device=='cpu' else False
    )

    test_csv = os.path.join(args.data.dataset_path, 'test.csv')
    test_spk_file = os.path.join(args.data.dataset_path, 'spk_info.csv')
    if os.path.isfile(test_csv) and os.path.isfile(test_spk_file):
        ds_test, test_spk_infos = get_datasets(test_csv, test_spk_file)

        loaders['test'] = DataLoader(
            AudioDataset(
                root_path=args.data.dataset_path,
                metadatas=ds_test,
                spk_infos=test_spk_infos,
                crop_duration=args.data.duration,
                hop_size=args.data.block_size,
                sampling_rate=args.data.sampling_rate,
                whole_audio=True,
                device=args.train.cache_device,
                use_aug=False,
                use_spk_embed=args.model.use_speaker_embed,
                units_only=args.train.only_u2c_stack),
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=True if args.train.cache_device=='cpu' else False
        )
    else:
        loaders['test'] = None

    return loaders