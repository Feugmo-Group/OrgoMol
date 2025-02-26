import os
import pandas as pd
from tqdm import tqdm
import re
import numpy as np
import logging
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


#read zmat
#turn into input string/sequence
#tokenizer
#['[BOND]', 'H-C', 1.09195, '[BOND]', 'H-C', 1.09195, '[ANGLE]', 'H-H', 109.46841, '[BOND]', 'H-C', 1.09195, '[ANGLE]', 'H-H', 109.47132, '[DIHEDRAL]', 'H-H', 119.99972, '[BOND]', 'H-H', 1.78316, '[ANGLE]', 'H-H', 60.00038, '[DIHEDRAL]', 'H-H', -70.52807]

def typeSecond(line,atomNum,atomDict):
    bondAtom = int(line[1])
    return ['[BOND]',f"{line[0]}-{atomDict[bondAtom]}",float(line[2])]
    
def typeThird(line,atomNum,atomDict): 
    bondAtom = int(line[1])
    angleAtom = int(line[3])
    return ['[BOND]',f"{line[0]}-{atomDict[bondAtom]}",float(line[2]),'[ANGLE]', f"{line[0]}-{atomDict[angleAtom]}",float(line[4])] 

def typeFourth(line,atomNum,atomDict):
    bondAtom = int(line[1])
    angleAtom = int(line[3])
    dihederalAtom = int(line[5])
    return ['[BOND]',f"{line[0]}-{atomDict[bondAtom]}",float(line[2]),'[ANGLE]', f"{line[0]}-{atomDict[angleAtom]}",float(line[4]),'[DIHEDRAL]',f"{line[0]}-{atomDict[dihederalAtom]}",float(line[6])]

class ValueNormalizer:
    def __init__(self):
        self.bond_stats = {'mean': None, 'std': None}
        self.angle_stats = {'mean': None, 'std': None}
        self.dihedral_stats = {'mean': None, 'std': None}
        self.eps = 1e-8
    
    def fit(self, values):
        """Compute normalization statistics for each type of value"""
        if isinstance(values, torch.Tensor):
            values = values.numpy()
            
        # Separate different types of values
        bond_values = values[:, ::3]      # Every 3rd value starting from 0
        angle_values = values[:, 1::3]    # Every 3rd value starting from 1
        dihedral_values = values[:, 2::3] # Every 3rd value starting from 2
        
        # Remove padding values (zeros) before computing stats
        bond_values = bond_values[bond_values != 0]
        angle_values = angle_values[angle_values != 0]
        dihedral_values = dihedral_values[dihedral_values != 0]
        
        # Compute statistics
        self.bond_stats['mean'] = float(np.mean(bond_values))
        self.bond_stats['std'] = float(np.std(bond_values) + self.eps)
        
        self.angle_stats['mean'] = float(np.mean(angle_values))
        self.angle_stats['std'] = float(np.std(angle_values) + self.eps)
        
        self.dihedral_stats['mean'] = float(np.mean(dihedral_values))
        self.dihedral_stats['std'] = float(np.std(dihedral_values) + self.eps)
        
        return self
    
    def transform(self, values):
        """Apply z-score normalization to values"""
        if isinstance(values, torch.Tensor):
            values = values.numpy()
        
        normalized = values.copy()
        
        # Only normalize non-zero values (ignore padding)
        bond_mask = values[:, ::3] != 0
        angle_mask = values[:, 1::3] != 0
        dihedral_mask = values[:, 2::3] != 0
        
        # Normalize each type of value separately
        normalized[:, ::3][bond_mask] = ((values[:, ::3][bond_mask] - self.bond_stats['mean']) 
                                       / self.bond_stats['std'])
        normalized[:, 1::3][angle_mask] = ((values[:, 1::3][angle_mask] - self.angle_stats['mean']) 
                                         / self.angle_stats['std'])
        normalized[:, 2::3][dihedral_mask] = ((values[:, 2::3][dihedral_mask] - self.dihedral_stats['mean']) 
                                            / self.dihedral_stats['std'])
        
        if isinstance(values, torch.Tensor):
            normalized = torch.from_numpy(normalized).float()
        
        return normalized
    
    def inverse_transform(self, normalized_values):
        """Convert normalized values back to original scale"""
        if isinstance(normalized_values, torch.Tensor):
            normalized_values = normalized_values.numpy()
            
        denormalized = normalized_values.copy()
        
        # Only denormalize non-zero values (ignore padding)
        bond_mask = normalized_values[:, ::3] != 0
        angle_mask = normalized_values[:, 1::3] != 0
        dihedral_mask = normalized_values[:, 2::3] != 0
        
        # Denormalize each type of value separately
        denormalized[:, ::3][bond_mask] = (normalized_values[:, ::3][bond_mask] * 
                                         self.bond_stats['std'] + self.bond_stats['mean'])
        denormalized[:, 1::3][angle_mask] = (normalized_values[:, 1::3][angle_mask] * 
                                           self.angle_stats['std'] + self.angle_stats['mean'])
        denormalized[:, 2::3][dihedral_mask] = (normalized_values[:, 2::3][dihedral_mask] * 
                                              self.dihedral_stats['std'] + self.dihedral_stats['mean'])
        
        if isinstance(normalized_values, torch.Tensor):
            denormalized = torch.from_numpy(denormalized).float()
            
        return denormalized

    def get_stats(self):
        """Return all normalization statistics"""
        return {
            'bond_stats': self.bond_stats,
            'angle_stats': self.angle_stats,
            'dihedral_stats': self.dihedral_stats
        }

    def load_stats(self, stats):
        """Load pre-computed statistics"""
        self.bond_stats = stats['bond_stats']
        self.angle_stats = stats['angle_stats']
        self.dihedral_stats = stats['dihedral_stats']

    def fit_transform(self, values):
        """Convenience method to fit and transform in one step"""
        return self.fit(values).transform(values)
    
class ZTokenizer():
    def __init__(self) -> None:
        self.vocab = {"[BOND]":1,"[ANGLE]":2,"[DIHEDRAL]":3,"[NUM]":5,"H-H":6,"H-C":7,"H-N":8,"H-O":9,"C-C":10,"C-N":11,"C-O":12,"N-N":13,"N-O":14,
                     "O-O":15,"N-C":11,"O-C":12,"O-N":14,"N-H":8,"C-H":7,"O-H":9,"[PAD]":0, "[CLS]":16}
        self.longest = 0
        self.num_token = 5

    def preProcess(self, zFile):
        """Process a Z-matrix file with debug information"""
        with open(zFile,"r") as myfile:
            atomNum = 1
            atomDict = {}
            processedString = []
            
            # Debug counters
            n_bonds = 0
            n_angles = 0
            n_dihedrals = 0
            
            for line in myfile.readlines(): 
                splitLine = line.split()
                
                if len(splitLine) == 1:
                    atomDict[atomNum] = line[0]
                    
                elif len(splitLine) == 3:
                    try:
                        atomDict[atomNum] = splitLine[0]
                        bond_value = float(splitLine[2])
                        processedString.extend(['[BOND]', f"{splitLine[0]}-{atomDict[int(splitLine[1])]}", bond_value])
                        n_bonds += 1
                    except (ValueError, IndexError) as e:
                        logging.error(f"Error processing bond in line {splitLine}: {e}")
                        
                elif len(splitLine) == 5:
                    try:
                        atomDict[atomNum] = splitLine[0]
                        bond_value = float(splitLine[2])
                        angle_value = float(splitLine[4])
                        processedString.extend([
                            '[BOND]', f"{splitLine[0]}-{atomDict[int(splitLine[1])]}", bond_value,
                            '[ANGLE]', f"{splitLine[0]}-{atomDict[int(splitLine[3])]}", angle_value
                        ])
                        n_bonds += 1
                        n_angles += 1
                    except (ValueError, IndexError) as e:
                        logging.error(f"Error processing bond/angle in line {splitLine}: {e}")
                        
                elif len(splitLine) == 7:
                    try:
                        atomDict[atomNum] = splitLine[0]
                        bond_value = float(splitLine[2])
                        angle_value = float(splitLine[4])
                        dihedral_value = float(splitLine[6])
                        processedString.extend([
                            '[BOND]', f"{splitLine[0]}-{atomDict[int(splitLine[1])]}", bond_value,
                            '[ANGLE]', f"{splitLine[0]}-{atomDict[int(splitLine[3])]}", angle_value,
                            '[DIHEDRAL]', f"{splitLine[0]}-{atomDict[int(splitLine[5])]}", dihedral_value
                        ])
                        n_bonds += 1
                        n_angles += 1
                        n_dihedrals += 1
                    except (ValueError, IndexError) as e:
                        logging.error(f"Error processing bond/angle/dihedral in line {splitLine}: {e}")
                
                atomNum += 1
            
            # Debug information
            logging.debug(f"File {zFile} statistics:")
            logging.debug(f"Total bonds processed: {n_bonds}")
            logging.debug(f"Total angles processed: {n_angles}")
            logging.debug(f"Total dihedrals processed: {n_dihedrals}")
            
            # Check for numeric values
            numeric_values = [x for x in processedString if isinstance(x, float)]
            if numeric_values:
                bonds = numeric_values[::3]
                angles = numeric_values[1::3]
                dihedrals = numeric_values[2::3]
                
                if bonds:
                    logging.debug(f"Bond range: [{min(bonds):.4f}, {max(bonds):.4f}]")
                if angles:
                    logging.debug(f"Angle range: [{min(angles):.4f}, {max(angles):.4f}]")
                if dihedrals:
                    logging.debug(f"Dihedral range: [{min(dihedrals):.4f}, {max(dihedrals):.4f}]")
            
            if len(processedString) > self.longest:
                self.longest = len(processedString)
                
            return processedString

    def pad(self, input):
        """Pad input while preserving numeric values"""
        # Store original numeric values and their positions
        numeric_values = [(i, x) for i, x in enumerate(input) if isinstance(x, float)]
        
        # Pad with "[PAD]" tokens
        while len(input) < self.longest:
            input.append("[PAD]")
        
        # Verify numeric values weren't affected
        for idx, val in numeric_values:
            if not isinstance(input[idx], float) or input[idx] != val:
                logging.error(f"Numeric value at position {idx} was modified during padding")
                logging.error(f"Original: {val}, Current: {input[idx]}")
        
        return input

    def encode(self, input):
        """Encode input with debug information"""
        encoded_tokens = []
        numerical_values = []
        
        # Track numeric values for debugging
        n_numeric = 0
        numeric_positions = []
        
        for i, token in enumerate(input):
            if isinstance(token, str):
                if token not in self.vocab:
                    logging.error(f"Unknown token encountered: {token}")
                encoded_tokens.append(self.vocab[token])
                numerical_values.append(0.0)
            elif isinstance(token, float):
                n_numeric += 1
                numeric_positions.append(i)
                encoded_tokens.append(self.num_token)
                numerical_values.append(token)
            else:
                logging.error(f"Unexpected token type at position {i}: {type(token)}")
        
        # Debug information
        if n_numeric > 0:
            values_array = np.array([x for x in numerical_values if x != 0.0])
            logging.debug(f"Encoded {n_numeric} numeric values")
            logging.debug(f"Numeric value stats - Mean: {np.mean(values_array):.4f}, "
                        f"Std: {np.std(values_array):.4f}")
            logging.debug(f"Numeric value range: [{np.min(values_array):.4f}, {np.max(values_array):.4f}]")
        
        return np.array(encoded_tokens), np.array(numerical_values)

    def decode(self, tokens, values):
        """Decode with validation"""
        decoded = []
        for token, value in zip(tokens, values):
            if token == self.num_token:
                if value == 0:
                    logging.warning("Found numeric token with zero value")
                decoded.append(value)
            else:
                try:
                    keyList = list(self.vocab.keys())
                    valList = list(self.vocab.values())
                    position = valList.index(token)
                    decoded.append(keyList[position])
                except ValueError:
                    logging.error(f"Unknown token encountered during decoding: {token}")
                    decoded.append("[UNK]")
        return decoded

    def tokenize(self, text):
        """Tokenize with validation"""
        raw_string = self.preProcess(text)
        
        # Debug: Check raw string contents
        numeric_count = sum(1 for x in raw_string if isinstance(x, float))
        logging.debug(f"Raw string contains {numeric_count} numeric values")
        
        padded_string = self.pad(raw_string)
        tokens, values = self.encode(padded_string)
        
        # Final validation
        if np.all(values == 0):
            logging.error("All values are zero after encoding!")
        if np.all(values == 1):
            logging.error("All values are one after encoding!")
            
        return tokens, values
    
class ZMatDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        tokens, values = self.texts[idx]
        label = self.labels[idx]
        return torch.tensor(tokens), torch.tensor(values), torch.tensor(label)

def createDataLoaders(tokenizer, data, batch_size, property_name, normalize=False, normalizer='z_norm'):
    texts, labels = process_data(data, property_name, tokenizer)
    
    if normalize:
        labels = normalize_labels(labels, normalizer)
    
    dataset = ZMatDataset(texts, labels)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader

def normalize_labels(labels, normalizer):
    if normalizer == 'z_norm':
        return (labels - np.mean(labels)) / np.std(labels)
    elif normalizer == 'mm_norm':
        return (labels - np.min(labels)) / (np.max(labels) - np.min(labels))
    elif normalizer == 'ls_norm':
        return np.log1p(labels)
    else:
        return labels


def tokenize(zPath):
    tok = ZTokenizer()
    normalizer = ValueNormalizer()
    tokens_list = []
    values_list = []
    excluded_files = 0
    total_files = 0

    # First pass: determine the maximum length
    for file in tqdm(os.listdir(zPath), desc="Determining max length"):
        raw_string = tok.preProcess(f"{zPath}/{file}")
        tok.pad(raw_string)  # This updates tok.longest

    # Second pass: tokenize and pad all sequences
    for file in tqdm(os.listdir(zPath), desc="Tokenizing and padding"):
        total_files += 1
        raw_string = tok.preProcess(f"{zPath}/{file}")
        padded_string = tok.pad(raw_string)
        tokens, values = tok.encode(padded_string)
        
        # Check for NaN in values
        if np.isnan(values).any():
            logging.warning(f"NaN detected in values for file: {file}. Excluding this file.")
            excluded_files += 1
            continue
        
        tokens_list.append(tokens)
        values_list.append(values)

    tokens_array = np.array(tokens_list)
    values_array = np.array(values_list)
    
    # Fit normalizer and normalize values
    normalizer.fit(values_array)
    normalized_values = normalizer.transform(values_array)

    logging.info(f"Total files processed: {total_files}")
    logging.info(f"Files excluded due to NaN: {excluded_files}")
    logging.info(f"Files included in the dataset: {len(tokens_list)}")
    
    # Log normalization statistics
    stats = normalizer.get_stats()
    logging.info("\nNormalization statistics:")
    logging.info(f"Bond stats - Mean: {stats['bond_stats']['mean']:.4f}, Std: {stats['bond_stats']['std']:.4f}")
    logging.info(f"Angle stats - Mean: {stats['angle_stats']['mean']:.4f}, Std: {stats['angle_stats']['std']:.4f}")
    logging.info(f"Dihedral stats - Mean: {stats['dihedral_stats']['mean']:.4f}, Std: {stats['dihedral_stats']['std']:.4f}")

    return tokens_array, normalized_values, tok.longest, normalizer



