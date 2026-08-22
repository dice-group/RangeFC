from torch.utils.data import DataLoader, random_split
import numpy as np
from copy import deepcopy
import pickle
import pandas as pd
import torch
import os

import random

import re

_Q_RE = re.compile(r"(Q\d+)$")
_P_RE = re.compile(r"(P\d+)$")

def _strip_brackets_url_lastseg(x: str) -> str:
    x = x.strip()
    if x.startswith("<") and x.endswith(">"):
        x = x[1:-1]
    return x.rstrip("/").split("/")[-1]

def normalize_ent_id(uri: str) -> str:
    seg = _strip_brackets_url_lastseg(uri)
    # Prefer trailing Qxxxx if present; else return last segment as-is
    m = _Q_RE.search(seg)
    return m.group(1) if m else seg

def normalize_rel_id(uri: str) -> str:
    seg = _strip_brackets_url_lastseg(uri)  # e.g., "Property:P729", "P729", "direct/P729"
    # Drop common prefixes like "Property:"
    if seg.startswith("Property:"):
        seg = seg.split(":", 1)[1]
    # Prefer trailing Pxxxx if present
    m = _P_RE.search(seg)
    return m.group(1) if m else seg


class Data:

    def __init__(self, args=None):

        data_dir = args.path_dataset_folder
        emb_typ = args.emb_type
        valid_ratio = args.valid_ratio
        selected_dataset_data_dir = data_dir+str(args.eval_dataset).lower()+"/"
        print(f" [DEBUG] Loading dataset from: {selected_dataset_data_dir}")
        #tmp_emb_folder = data_dir + str(args.eval_dataset).lower()+"/embeddings/"
        import os
        tmp_emb_folder = os.path.join(selected_dataset_data_dir, "embeddings", emb_typ, "")

        def _ensure_id_maps():
            import os
            # prefer new *map.tsv names if present, otherwise fallback to legacy names
            def choose(path_no_ext, tsv_name):
                p1 = os.path.join(selected_dataset_data_dir, tsv_name)
                p0 = os.path.join(selected_dataset_data_dir, path_no_ext)  # legacy (no ext)
                return p1 if os.path.exists(p1) else p0

            ent_map_path = choose("entities", "entities_map.tsv")
            rel_map_path = choose("relations", "relations_map.tsv")
            tim_map_path = choose("times", "times_map.tsv")

            maps_exist = all(os.path.exists(p) for p in [ent_map_path, rel_map_path, tim_map_path])
            if maps_exist:
                self.idx_ent_dict = self.get_ids_dict(ent_map_path)
                self.idx_rel_dict = self.get_ids_dict(rel_map_path)
                self.idx_time_dict = self.get_ids_dict(tim_map_path)

                # widen entity/rel keys so “<…/Q####>”, URLs, or bare IDs all resolve
                ent_widen = {}
                for k, v in list(self.idx_ent_dict.items()):
                    kk = k.strip("<>")
                    seg = kk.rstrip("/").split("/")[-1]
                    if seg != k:
                        ent_widen[seg] = v
                self.idx_ent_dict.update(ent_widen)

                rel_widen = {}
                for k, v in list(self.idx_rel_dict.items()):
                    kk = k.strip("<>")
                    seg = kk.rstrip("/").split("/")[-1]
                    norm = normalize_rel_id(seg)  # ensures bare P####
                    rel_widen[norm] = v
                    if seg.startswith("Property:"):
                        rel_widen[seg] = v
                self.idx_rel_dict.update(rel_widen)

                print(" [DEBUG] after-widen: ents=", len(self.idx_ent_dict),
                      "rels=", len(self.idx_rel_dict), "times=", len(self.idx_time_dict))
                print(" [DEBUG] sample ent keys:", list(self.idx_ent_dict.keys())[:5])
                print(" [DEBUG] sample rel keys:", list(self.idx_rel_dict.keys())[:5])
                return

            # Build maps by scanning raw train/test/valid (URI quintuples)
            train_raw = self.read_raw_lines(os.path.join(selected_dataset_data_dir + "train/", "train"))
            valid_raw = self.read_raw_lines(os.path.join(selected_dataset_data_dir + "valid/", "valid"))
            test_raw = self.read_raw_lines(os.path.join(selected_dataset_data_dir + "test/", "test"))
            all_raw = train_raw + valid_raw + test_raw
            self.entities = self.get_entities(all_raw)
            self.relations = self.get_relations(all_raw)
            self.times = self.get_times(all_raw)

            def _last(seg):
                s = seg.strip("<>")
                return s.rstrip("/").split("/")[-1]

            self.entities = [_last(s) for s, _, o, _, _ in all_raw] + [_last(o) for s, _, o, _, _ in all_raw]
            self.entities = sorted(set(self.entities))

            self.relations = sorted(set(
                [normalize_rel_id(p) for _, p, _, _, _ in all_raw]
            ))

            # Persist maps
            with open(selected_dataset_data_dir + "entities", "w") as f:
                for i,e in enumerate(self.entities): f.write(f"{i} {e}\n")
            with open(selected_dataset_data_dir + "relations", "w") as f:
                for i,r in enumerate(self.relations): f.write(f"{i} {r}\n")
            with open(selected_dataset_data_dir + "times", "w") as f:
                for i,t in enumerate(self.times): f.write(f"{i} {t}\n")
            # Reload as dicts in "ID -> index" form
            self.idx_ent_dict = self.get_ids_dict(selected_dataset_data_dir + "entities")
            self.idx_rel_dict = self.get_ids_dict(selected_dataset_data_dir + "relations")
            self.idx_time_dict = self.get_ids_dict(selected_dataset_data_dir + "times")

        def _load_embeddings():
            import os
            # helpers
            def exists(fname):
                return os.path.exists(os.path.join(tmp_emb_folder, fname))

            # --- Entities ---
            if exists("entity.npy"):
                self.emb_entities = self.get_embeddings(tmp_emb_folder, "entity.npy")
                # if your .npy contains reserved rows at top, trim here (optional)
                if len(self.emb_entities) != len(self.idx_ent_dict):
                    extra = len(self.emb_entities) - len(self.idx_ent_dict)
                    if extra > 0:
                        print(f"Trimming {extra} extra entity rows from .npy")
                        self.emb_entities = self.emb_entities[extra:]
                        if len(set(self.idx_ent_dict.values())) != len(self.emb_entity):
                            raise SystemExit("[FATAL] unique entity IDs != entity.npy rows")
                print(f"[DEBUG] loaded entity.npy: {len(self.emb_entities)} rows")
            elif exists("entity.pkl"):
                self.emb_entities = self.get_embeddings(tmp_emb_folder, "entity.pkl")
                if len(self.emb_entities) != len(self.idx_ent_dict):
                    extra = len(self.emb_entities) - len(self.idx_ent_dict)
                    if extra > 0:
                        print(f"Trimming {extra} extra entity rows from .pkl")
                        self.emb_entities = self.emb_entities[extra:]
                print(f"[DEBUG] loaded entity.pkl: {len(self.emb_entities)} rows")
            else:
                # CSV requires an order column; enforce map order
                order = [None] * len(self.idx_ent_dict)
                for iri, idx in self.idx_ent_dict.items():
                    order[idx] = iri
                self.emb_entities = self.get_embeddings_from_csv(tmp_emb_folder, "all_entities_embeddings_final.csv",
                                                                 order)
                print(f"[DEBUG] loaded all_entities_embeddings_final.csv: {len(self.emb_entities)} rows")

            # --- Relations ---
            if exists("relation.npy"):
                self.emb_relation = self.get_embeddings(tmp_emb_folder, "relation.npy")
                if len(self.emb_relation) != len(self.idx_rel_dict):
                    extra = len(self.emb_relation) - len(self.idx_rel_dict)
                    if extra > 0:
                        print(f"Trimming {extra} extra relation rows from .npy")
                        self.emb_relation = self.emb_relation[extra:]
                        uniq_rel_ids = len(set(self.idx_rel_dict.values()))
                        if uniq_rel_ids != len(self.emb_relation):
                            raise SystemExit(
                                f"[FATAL] unique relation IDs in map={uniq_rel_ids} "
                                f"but relation embedding rows={len(self.emb_relation)}. "
                                f"Load the matching file (e.g., relation5.npy) or fix the map."
                            )
                print(f"[DEBUG] loaded relation.npy: {len(self.emb_relation)} rows")
            elif exists("relation.pkl"):
                self.emb_relation = self.get_embeddings(tmp_emb_folder, "relation.pkl")
                if len(self.emb_relation) != len(self.idx_rel_dict):
                    extra = len(self.emb_relation) - len(self.idx_rel_dict)
                    if extra > 0:
                        print(f"Trimming {extra} extra relation rows from .pkl")
                        self.emb_relation = self.emb_relation[extra:]
                print(f"[DEBUG] loaded relation.pkl: {len(self.emb_relation)} rows")
            else:
                order = [None] * len(self.idx_rel_dict)
                for iri, idx in self.idx_rel_dict.items():
                    order[idx] = iri
                self.emb_relation = self.get_embeddings_from_csv(tmp_emb_folder, "all_relations_embeddings_final.csv",
                                                                 order)
                print(f"[DEBUG] loaded all_relations_embeddings_final.csv: {len(self.emb_relation)} rows")

            # --- Times (optional) ---
            if exists("time.npy"):
                self.emb_times = self.get_embeddings(tmp_emb_folder, "time.npy")
                print(f"[DEBUG] loaded time.npy: {len(self.emb_times)} rows")
            elif exists("time.pkl"):
                self.emb_times = self.get_embeddings(tmp_emb_folder, "time.pkl")
                print(f"[DEBUG] loaded time.pkl: {len(self.emb_times)} rows")
            else:
                self.emb_times = []

            self.num_entities = len(self.emb_entities)
            self.num_relations = len(self.emb_relation)
            self.num_times = len(self.idx_time_dict)  # range task uses buckets, not emb_times

        def _probe_split_for_misses(path, max_print=5):
            missing_h = missing_r = missing_t = 0
            ex_h, ex_r, ex_t = [], [], []
            lines = self.read_raw_lines(path)
            for ln in lines:
                parts = ln.split('\t')
                if len(parts) != 5:
                    continue
                h, r, t, y1, y2 = parts
                h_id = normalize_ent_id(h)
                r_id = normalize_rel_id(r)
                t_id = normalize_ent_id(t)
                if h_id not in self.idx_ent_dict and len(ex_h) < max_print:
                    ex_h.append(h_id);
                    missing_h += 1
                if r_id not in self.idx_rel_dict and len(ex_r) < max_print:
                    ex_r.append(r_id);
                    missing_r += 1
                if t_id not in self.idx_ent_dict and len(ex_t) < max_print:
                    ex_t.append(t_id);
                    missing_t += 1
            print(f"[DEBUG] Probe {path}: missing_h={missing_h}, missing_r={missing_r}, missing_t={missing_t}")
            if ex_h: print("  e.g. missing h:", ex_h)
            if ex_r: print("  e.g. missing r:", ex_r)
            if ex_t: print("  e.g. missing t:", ex_t)

        ids_only = args.ids_only


        # Quick workaround as we happen to have duplicate triples.
        # None if load complete data, otherwise load parts of dataset with folders in wrong directory.
        # emb_folder = ""

        if args.task == 'range-prediction':
            _ensure_id_maps()           # <-- maps exist before reading ranges
            _load_embeddings()          # <-- and embeddings too
            print("[DEBUG] Loading range-prediction 5-tuple dataset format...")

            print(" [DEBUG] map sizes:",
                  "ents=", len(self.idx_ent_dict),
                  "rels=", len(self.idx_rel_dict),
                  "times=", len(self.idx_time_dict))
            print(" [DEBUG] sample rel keys:", list(self.idx_rel_dict.keys())[:5])

            valid_path = os.path.join(selected_dataset_data_dir, "valid", "valid")
            train_path = os.path.join(selected_dataset_data_dir, "train", "train")
            test_path = os.path.join(selected_dataset_data_dir, "test", "test")
            _probe_split_for_misses(train_path)
            _probe_split_for_misses(valid_path)
            _probe_split_for_misses(test_path)

            #step1: read all files just to get triples
            #train_raw = self.read_raw_lines(selected_dataset_data_dir + "train/train")
            #test_raw = self.read_raw_lines(selected_dataset_data_dir + "test/test")
            #valid_raw = self.read_raw_lines(selected_dataset_data_dir + "valid/valid")

            #step2: extract/entities/relations/times from all splits
            #self.entities = self.get_entities(train_raw + test_raw + valid_raw)
            #self.relations = self.get_relations(train_raw + test_raw + valid_raw)
            #self.times = self.get_times(train_raw + test_raw + valid_raw)

            #step3: build index mapping
            #self.idx_ent_dict = {e.rsplit("/", 1)[-1]: i for i, e in enumerate(self.entities)}
            #self.idx_rel_dict = {r.rsplit("/", 1)[-1]: i for i, r in enumerate(self.relations)}
            #self.idx_time_dict = {str(int(float(t))): i for i, t in enumerate(self.times)}

            #step4: load indexed data
            self.idx_train_set = self.load_range_data(selected_dataset_data_dir + "train/train")
            self.idx_test_set = self.load_range_data(selected_dataset_data_dir + "test/test")
            self.idx_valid_set = self.load_range_data(selected_dataset_data_dir + "valid/valid")

            print(f"[DEBUG] Train entries: {len(self.idx_train_set)}")
            print(f"[DEBUG] Test entries: {len(self.idx_test_set)}")
            print(f"[DEBUG] Valid entries: {len(self.idx_valid_set)}")

            # === Per-relation priors (midpoint & duration) from TRAIN, in index space ===
            import numpy as np
            # self.idx_train_set rows are [s_idx, r_idx, t_idx, y1_idx, y2_idx]
            train_arr = np.asarray(self.idx_train_set, dtype=np.int64)
            r_train = train_arr[:, 1]
            y1 = train_arr[:, 3].astype(np.float32)
            y2 = train_arr[:, 4].astype(np.float32)
            mid = 0.5 * (y1 + y2)
            dur = np.maximum(0.0, y2 - y1)

            R = len(self.idx_rel_dict.values())  # number of (unique id) relations; safer to use:
            R = len(set(self.idx_rel_dict.values()))

            prior_m = np.zeros((R,), dtype=np.float32)
            prior_d = np.zeros((R,), dtype=np.float32)
            for rid in range(R):
                mask = (r_train == rid)
                if not mask.any():
                    # fallback: global median if rid unseen in train
                    prior_m[rid] = float(np.median(mid))
                    prior_d[rid] = float(np.median(dur))
                else:
                    prior_m[rid] = float(np.median(mid[mask]))
                    prior_d[rid] = float(np.median(dur[mask]))

            # store for the model
            self.relation_prior_mid_idx = prior_m  # float array length R (indices)
            self.relation_prior_dur_idx = prior_d  # float array length R (indices)

            print(f"[DEBUG] per-relation priors (index): mid[0..4]={prior_m[:5]}, dur[0..4]={prior_d[:5]}")



        #if args.model == "KGE-only":
         #   self.process_KGE_only_data(selected_dataset_data_dir, args)

        elif ids_only == False:
            self.train_set_time_final = list((self.load_data(selected_dataset_data_dir + "train/", data_type="train")))
            self.test_set_time_final = list((self.load_data(selected_dataset_data_dir + "test/", data_type="test")))
            self.valid_set_time_final = list((self.load_data(selected_dataset_data_dir + "valid/", data_type="valid")))

        #    print(f"Train loaded: {len(self.train_set_time_final)} records")
        #    print(f"Valid loaded: {len(self.valid_set_time_final)} records")
        #    print(f"Test loaded: {len(self.test_set_time_final)} records")

            #self.test_set_time_final, self.valid_set_time_final = self.generate_test_valid_set(self,self.test_set_time_final, valid_ratio)
            #if args.include_veracity == True:
                # factcheck predictions on train and test data. Should be added here before: 'data_TP/dbpedia124k/factcheck_veracity_scores/train_pred'
             #   self.train_set_pred = list((self.load_data(selected_dataset_data_dir + "train/", data_type="train_v_scores.txt", pred=True)))
             #   self.test_set_pred = list((self.load_data(selected_dataset_data_dir + "test/", data_type="test_v_scores.txt", pred=True)))

              #  self.test_set_pred, self.valid_set_pred = self.generate_test_valid_set(self, self.test_set_pred, valid_ratio)

            # generate test and validation sets
            # self.test_set, self.valid_set = self.generate_test_valid_set(self, self.test_set)



            ###########################################################################################################
            ##########################################################################################################
            ##################SENTENCE WORLD###########################################################
            #my editing
            try:
                self.emb_sentences_train = pd.read_csv(selected_dataset_data_dir + "train/" + "trainSE.csv", sep=",").iloc[:, 3:]
            except FileNotFoundError:
                print("ℹ️  trainSE.csv not found; skipping sentence embeddings.")
                self.emb_sentences_train = pd.DataFrame()
            try:
                self.emb_sentences_test = pd.read_csv(selected_dataset_data_dir + "test/" + "testSE.csv", sep=",").iloc[:, 3:]
            except FileNotFoundError:
                print("ℹ️  testSE.csv not found; skipping sentence embeddings.")
                self.emb_sentences_test = pd.DataFrame()
            #if you split off a validataion slice of the sentence embeddings:
            if not self.emb_sentences_test.empty:
                self.emb_sentences_test, self.emb_sentences_valid = self.generate_test_valid_sentence_set(self, self.emb_sentences_test, valid_ratio)
            else:
                self.emb_sentences_valid = pd.DataFrame()
            #############################################################################################################
            ##############################################################################################################
            # get all entities and relations
            # self.data = self.train_set + list(self.test_set) + list(self.valid_set)
            self.data = self.train_set_time_final + self.test_set_time_final + self.valid_set_time_final
            self.entities = self.get_entities(self.data)

                   # self.relations = list(set(self.get_relations(self.train_set) + self.get_relations(self.test_set)))
            self.relations = self.get_relations(self.data)

            self.times = self.get_times(self.data)
            self.save_all_resources(self.entities, selected_dataset_data_dir, is_entity=True)

            #--------------- my edititing -----------------
            # write the entity ID, relation ID, time ID maps
            # (so that get_ids_dict can load them back in as exact integers)
            with open(selected_dataset_data_dir + "entities", "w") as f:
                for idx, ent in enumerate(self.entities):
                    f.write(f"{idx} {ent}\n")
            with open(selected_dataset_data_dir + "relations", "w") as f:
                for idx, rel in enumerate(self.relations):
                    f.write(f"{idx} {rel}\n")
            with open(selected_dataset_data_dir + "times", "w") as f:
                for idx, tim in enumerate(self.times):
                    f.write(f"{idx} {tim}\n")
            # --------------- my edititing -----------------

            # self.save_all_resources(self.relations, selected_dataset_data_dir, is_entity=False)
            # exit(1)
            self.idx_ent_dict = dict()
            self.idx_rel_dict = dict()
            self.idx_time_dict = dict()

            # Generate integer mapping
            #for i in self.entities:
            #    self.idx_ent_dict[i.replace("<http://dbpedia.org/resource/", "")[:-1]] = len(self.idx_ent_dict)
            #for i in self.relations:
            #    self.idx_rel_dict[i.replace("<http://dbpedia.org/ontology/", "")[:-1]] = len(self.idx_rel_dict)
            #for i in self.times:
            #    self.idx_time_dict[i] = len(self.idx_time_dict)

            #-----------------my editing-----------------
            # Entities
            for uri in self.entities:
                ent_id = normalize_ent_id(str(uri))
                self.idx_ent_dict[ent_id] = len(self.idx_ent_dict)

            # Relations
            for uri in self.relations:
                rel_id = normalize_rel_id(str(uri))
                self.idx_rel_dict[rel_id] = len(self.idx_rel_dict)

            # Times
            for t in self.times:
                # keep as strings of integer years
                self.idx_time_dict[str(int(float(t)))] = len(self.idx_time_dict)

            # -----------------my editing-----------------

            if args.include_veracity == True:
                self.copaal_veracity_train = self.get_veracity_data(self, self.train_set_pred)
                self.copaal_veracity_test = self.get_veracity_data(self, self.test_set_pred)
                self.copaal_veracity_valid = self.get_veracity_data(self, self.valid_set_pred)


            if str(args.model).__contains__("temporal"):
                self.emb_entities = self.get_embeddings(tmp_emb_folder + emb_typ + '/','entity.pkl')
                self.emb_relation = self.get_embeddings(tmp_emb_folder + emb_typ + '/','relation.pkl')
                self.emb_time = self.get_embeddings(tmp_emb_folder+emb_typ+'/','time.pkl')
            else:
                self.emb_entities = self.get_embeddings(tmp_emb_folder + emb_typ + '/', 'all_entities_embeddings_final.csv')
                self.emb_relation = self.get_embeddings(tmp_emb_folder + emb_typ + '/',
                                                        'all_relations_embeddings_final.csv')

            self.num_entities = len(self.emb_entities)
            assert len(self.emb_entities) == len(self.idx_ent_dict), "Entity count mismatch"
            self.num_relations = len(self.emb_relation)
            self.num_times = 0
            if str(args.model).__contains__("temporal"):
                self.num_times = len(self.emb_time)

            if args.negative_triple_generation =="corrupted-time-based": # we have to duplicate the sentences because only time is currupted in this case..
                # TODO for later
                # concatinating sentence embeddings assumes that the first half are true examples.
                self.train_set_time_final, count = self.generate_negative_triples(self.train_set_time_final,type=args.negative_triple_generation)
                self.emb_sentences_train = self.emb_sentences_train[:count]
                self.emb_sentences_train = pd.concat([self.emb_sentences_train, self.emb_sentences_train],
                                                     ignore_index=True)
                self.valid_set_time_final, count = self.generate_negative_triples(self.valid_set_time_final,type=args.negative_triple_generation)
                self.emb_sentences_valid = self.emb_sentences_valid[:count]
                self.emb_sentences_valid = pd.concat([self.emb_sentences_valid, self.emb_sentences_valid],
                                                     ignore_index=True)

                self.test_set_time_final, count = self.generate_negative_triples(self.test_set_time_final,type=args.negative_triple_generation)
                self.emb_sentences_test = self.emb_sentences_test[:count]
                self.emb_sentences_test = pd.concat([self.emb_sentences_test, self.emb_sentences_test],
                                                     ignore_index=True)

            elif args.negative_triple_generation == "False":
                self.train_set_time_final = self.generate_only_true_triples(self.train_set_time_final)
                self.valid_set_time_final = self.generate_only_true_triples(self.valid_set_time_final)
                self.test_set_time_final = self.generate_only_true_triples(self.test_set_time_final)

            #print(f"[DEBUG RAW] first 3 TRAIN quintuples:", self.train_set_time_final[:3])
            #print(f"[DEBUG RAW] first 3 TEST quintuples:", self.test_set_time_final[:3])

            from collections import defaultdict
            subject_groups = defaultdict(list)
            for s, p, o, time, label in self.train_set_time_final:
                subject_groups[s].append((p, o, time, label))
            self.train_range_data = []
            for subj, recs in subject_groups.items():
                if len(recs) < 2:
                    continue
                for i in range(len(recs)):
                    p1, o1, y1, _ = recs[i]
                    for j in range(i+1, len(recs)):
                        _, _, y2, _ = recs[j]
                        self.train_range_data.append((subj, p1, o1, y1, y2))
            print(f"[DEBUG] grouped {len(self.train_range_data)}train-range tuples")
            print(f"[DEBUG] sample:", self.train_range_data[:5])

            self.idx_train_set = []
            for sent_i, (subj, pred, obj, y1, y2) in enumerate(self.train_range_data):
                # 2.1 clean your URI → just take the final ID
                s_key = subj.rsplit("/", 1)[-1]
                p_key = pred.rsplit("/", 1)[-1]
                o_key = obj.rsplit("/", 1)[-1]

                #2.2 map to your dictionary
                try:
                    idx_s = self.idx_ent_dict[s_key]
                    idx_p = self.idx_rel_dict[p_key]
                    idx_o = self.idx_ent_dict[o_key]
                    idx_y1 = self.idx_time_dict[str(y1)]
                    idx_y2 = self.idx_time_dict[str(y2)]
                except KeyError as e:
                    print(f"[DEBUG-OOB] Missing key {e} for pair {(s_key, p_key, o_key, y1, y2)}")
                    continue
                #2.3 append exactly two time-indices instead of one
                self.idx_train_set.append([idx_s, idx_p, idx_o, idx_y1, idx_y2])

            #2.4 sanity-check
            print(f"[DEBUG] Build {len(self.idx_train_set)} indexed train-range entries")
            print(f"[DEBUG] sample indexed entries:", self.idx_train_set[:5])
            #i = 0
            #sent_i = 0
            #len_train = len(self.train_set_time_final)
            """
            for (s, p, o, time, label) in self.train_set_time_final:
                #s = str(s).replace("<http://dbpedia.org/resource/", "")[:-1]
                #s = str(s).replace("<http://dbpedia.org/resource/", "") \
                #    .replace("<http://www.wikidata.org/entity/", "") \
                #    .strip(">")  # Remove trailing ">" safely
                #p = str(p).replace("<http://dbpedia.org/ontology/", "")[:-1].replace("Of","")
                #o = str(o).replace("<http://dbpedia.org/resource/", "")[:-1]
                #o = str(o).replace("<http://dbpedia.org/resource/", "") \
                #    .replace("<http://www.wikidata.org/entity/", "") \
                #    .strip(">")  # Remove trailing ">" safely

                # -----------------my editing-----------------

                #subject
                s = str(s)
                if s.startswith("<") and s.endswith(">"):
                    s = s[1:-1]
                s = s.rstrip("/").split("/")[-1]

                #predicate
                p = str(p)
                if p.startswith("<") and p.endswith(">"):
                    p = p[1:-1]
                p = p.rstrip("/").split("/")[-1]

                #object
                o = str(o)
                if o.startswith("<") and o.endswith(">"):
                    o = o[1:-1]
                o = o.rstrip("/").split("/")[-1]

                # -----------------my editing-----------------

                if self.idx_ent_dict.keys().__contains__(s) and self.idx_rel_dict.keys().__contains__(p) and self.idx_ent_dict.keys().__contains__(o):
                    idx_s, idx_p, idx_o, idx_t,  label = min(self.idx_ent_dict[s], self.num_entities - 1) , self.idx_rel_dict[p], self.idx_ent_dict[o], self.idx_time_dict[time], label
                    if label == 'True' or label == 1:
                        label = 1
                    else:
                        label = 0
                    ver = i
                    #     this is to just to make sure if any time is not the same even after randomaly shuffling so increment by 1
                    if args.negative_triple_generation =="corrupted-time-based" and (i >= int(len_train/2)):
                        item = self.idx_train_set.__getitem__(i-int(len_train/2))
                        if ((item[0] != int(idx_s)) or (item[1] != int(idx_p)) or (item[2]!=int(idx_o))):
                            print("serious problem, please check")
                            exit(1)
                        if ((item[0]==int(idx_s)) and (item[1]==int(idx_p)) and (item[2]==int(idx_o)) and (item[3]==int(idx_t))):
                            idx_t = (int(idx_t) + 1) if ((int(idx_t)+1) < self.num_times) else 0
                        ver = item[4]
                    self.idx_train_set.append([int(idx_s), int(idx_p), int(idx_o),int(idx_t), sent_i, ver, label])
                else:
                    print("check:"+s + ","+o)
                i = i + 1
                sent_i = sent_i + 1

                #-----------------------my editing----------------------
                s_clean = s.replace("<http://dbpedia.org/resource/", "")
                if s_clean not in self.idx_ent_dict:
                    print(f"❌ Missing entity in idx_ent_dict: {s_clean}")
                    raise ValueError("Entity not in embeddings")
                # -----------------------my editing----------------------
            """

            self.idx_valid_set = []
            j = 0
            sent_i = 0
            len_valid = len(self.valid_set_time_final)
            for (s, p, o, time, label) in self.valid_set_time_final:

                #----------------my editing------------------

                #s = str(s).replace("<http://dbpedia.org/resource/", "")[:-1]
                #p = str(p).replace("<http://dbpedia.org/ontology/", "")[:-1].replace("Of","")
                #o = str(o).replace("<http://dbpedia.org/resource/", "")[:-1]

                # subject
                s = str(s)
                if s.startswith("<") and s.endswith(">"):
                    s = s[1:-1]
                s = s.rstrip("/").split("/")[-1]

                # predicate
                p = str(p)
                if p.startswith("<") and p.endswith(">"):
                    p = p[1:-1]
                p = p.rstrip("/").split("/")[-1]

                # object
                o = str(o)
                if o.startswith("<") and o.endswith(">"):
                    o = o[1:-1]
                o = o.rstrip("/").split("/")[-1]

                # -----------------my editing-----------------

                if self.idx_ent_dict.keys().__contains__(s) and  self.idx_rel_dict.keys().__contains__(p) and self.idx_ent_dict.keys().__contains__(o):
                    idx_s, idx_p, idx_o, idx_t, label = min(self.idx_ent_dict[s], self.num_entities - 1) , self.idx_rel_dict[p], self.idx_ent_dict[o],self.idx_time_dict[time], label
                    if label == 'True' or label == 1:
                        label = 1
                    else:
                        label = 0
                    ver = j
                    #     this is to check if any time is same even after randomaly shuffling so increment by 1
                    if args.negative_triple_generation =="corrupted-time-based" and (j >= int(len_valid/2)):
                        item = self.idx_valid_set.__getitem__(j-int(len_valid/2))
                        if ((item[0] != int(idx_s)) or (item[1] != int(idx_p)) or (item[2]!=int(idx_o))):
                            print("serious problem, please check")
                            exit(1)
                        if ((item[0]==int(idx_s)) and (item[1]==int(idx_p)) and (item[2]==int(idx_o)) and (item[3]==int(idx_t))):
                            idx_t = (int(idx_t) + 1) if ((int(idx_t)+1) < self.num_times) else 0
                        ver = item[4]
                    self.idx_valid_set.append([int(idx_s), int(idx_p), int(idx_o),int(idx_t), sent_i, ver, label])
                else:
                    print("check:" + s + "," + o)
                j = j + 1
                sent_i = sent_i + 1

            self.idx_test_set = []
            k = 0
            sent_i = 0
            len_test = len(self.test_set_time_final)
            for (s, p, o, time, label) in self.test_set_time_final:

                #---------------my editing----------------

                #s = str(s).replace("<http://dbpedia.org/resource/", "")[:-1]
                #p = str(p).replace("<http://dbpedia.org/ontology/", "")[:-1].replace("Of","")
                #o = str(o).replace("<http://dbpedia.org/resource/", "")[:-1]

                # subject
                s = str(s)
                if s.startswith("<") and s.endswith(">"):
                    s = s[1:-1]
                s = s.rstrip("/").split("/")[-1]

                # predicate
                p = str(p)
                if p.startswith("<") and p.endswith(">"):
                    p = p[1:-1]
                p = p.rstrip("/").split("/")[-1]

                # object
                o = str(o)
                if o.startswith("<") and o.endswith(">"):
                    o = o[1:-1]
                o = o.rstrip("/").split("/")[-1]

                # -----------------my editing-----------------

                if self.idx_ent_dict.keys().__contains__(s) and  self.idx_rel_dict.keys().__contains__(p) and self.idx_ent_dict.keys().__contains__(o):
                    idx_s, idx_p, idx_o, idx_t, label = min(self.idx_ent_dict[s], self.num_entities - 1) , self.idx_rel_dict[p], self.idx_ent_dict[o],self.idx_time_dict[time], label
                    if label == 'True' or label == 1:
                        label = 1
                    else:
                        label = 0
                    ver = k
                    #     this is to check if any time is same even after randomaly shuffling so increment by 1
                    if args.negative_triple_generation =="corrupted-time-based" and (k >= int(len_test/2)):
                        item = self.idx_test_set.__getitem__(k-int(len_test/2))
                        if ((item[0] != int(idx_s)) or (item[1] != int(idx_p)) or (item[2]!=int(idx_o))):
                            print("serious problem, please check")
                            exit(1)
                        if ((item[0]==int(idx_s)) and (item[1]==int(idx_p)) and (item[2]==int(idx_o)) and (item[3]==int(idx_t))):
                            idx_t = (int(idx_t) + 1) if ((int(idx_t)+1) < self.num_times) else 0
                        ver = item[4]
                    self.idx_test_set.append([int(idx_s), int(idx_p), int(idx_o),int(idx_t), sent_i, ver, label])
                else:
                    print("check:" + s + "," + o)
                k = k + 1
                sent_i = sent_i + 1

            from collections import Counter

            def dbg_split(name, idx_list):
                keys = [tuple(rec[:3]) for rec in idx_list]
                ctr = Counter(keys)
                total_groups = len(ctr)
                paired_groups = sum(1 for v in ctr.values() if v >= 2)
              #  print(f" {name} #records: {len(idx_list)}, #groups: {total_groups}, #with>=2 time-points: {paired_groups}")

            dbg_split("TRAIN", self.idx_train_set)
            dbg_split("VALID", self.idx_valid_set)
            dbg_split("TEST", self.idx_test_set)

            from collections import defaultdict

            def dbg_split_by_subject(name, idx_list):
                print(f"checking {name} set")
                subject_groups = defaultdict(list)
                for rec in idx_list:
                    subject_id = rec[0]
                    subject_groups[subject_id].append(rec)

                total_groups = len(subject_groups)
                group_with_2_or_more = sum(1 for grp in subject_groups.values() if len(grp) >= 2)

              #  print(f"{name} #records: {len(idx_list)}, #unique subjects: {total_groups}, #subjects with >= 2 records: {group_with_2_or_more} ")

            dbg_split_by_subject("TRAIN", self.idx_train_set)
            dbg_split_by_subject("VALID", self.idx_valid_set)
            dbg_split_by_subject("TEST", self.idx_test_set)

            # ——— STEP: generate paired indices for range‐prediction ———
            from utils_TP.dataset_classes import make_pairs

            #print(" About to make_pairs on train; #train_idx:", len(self.idx_train_set))
            self.paired_train_idx = make_pairs(self.idx_train_set)
            #print(" Made paired_train_idx; length:", len(self.paired_train_idx))

            #print(" about to make_pairs on valid; #valid_idx:", len(self.idx_valid_set))
            self.paired_valid_idx = make_pairs(self.idx_valid_set)
            #print(" Made paired_valid_idx; length:", len(self.paired_valid_idx))

            #print(" about to make_pairs on test; #test_idx:", len(self.idx_test_set))
            self.paired_test_idx = make_pairs(self.idx_test_set)
            #print(" Made paired_test_idx; length:", len(self.paired_test_idx))
        else:

            # --------------- my edititing -----------------
            self.idx_ent_dict = self.get_ids_dict(selected_dataset_data_dir+"entities")
            self.idx_rel_dict = self.get_ids_dict(selected_dataset_data_dir+"relations")
            self.idx_time_dict = self.get_ids_dict(selected_dataset_data_dir+"times")

            _load_embeddings()
            self.idx_train_set = self.load_id_split(selected_dataset_data_dir + "train/train")
            self.idx_test_set = self.load_id_split(selected_dataset_data_dir + "test/test")
            self.idx_valid_set = self.load_id_split(selected_dataset_data_dir + "valid/valid")


            # --------------- my edititing -----------------

            #self.emb_entities = self.get_embeddings( tmp_emb_folder + emb_typ + '/', 'entity.pkl')
            #self.emb_relation = self.get_embeddings( tmp_emb_folder + emb_typ + '/', 'relation.pkl')
            #self.emb_time = self.get_embeddings( tmp_emb_folder + emb_typ + '/', 'time.pkl')

            import os
            # ── Entities ───────────────────────────────────────────────────────────
            ent_pkl = os.path.join(tmp_emb_folder, emb_typ, 'entity.pkl')
            print("  >> ent_pkl exists? ", os.path.exists(ent_pkl))
            if os.path.exists(ent_pkl):
                #self.emb_entities = self.get_embeddings(tmp_emb_folder + emb_typ + '/', 'entity.pkl')
                #print(f"  >> #mapped entities = {len(self.idx_ent_dict)}, #loaded embeddings = {len(self.emb_entities)}")
                # load the full embedding (including any reserved tokens)
                full_ent_emb = self.get_embeddings(tmp_emb_folder + emb_typ + '/', 'entity.pkl')
            #    print(f" >> #mapped entities = {len(self.idx_ent_dict)}, #loaded embeddings = {len(full_ent_emb)}")
                # if there are extra rows, assume they’re the first N reserved tokens; drop them:
                if len(full_ent_emb) > len(self.idx_ent_dict):
                    extra = len(full_ent_emb) - len(self.idx_ent_dict)
                    print(f"Trimming off {extra} reserved‐token embeddings")
                    full_ent_emb = full_ent_emb[extra:]
                self.emb_entities = full_ent_emb
            else:
                # load from CSV and enforce the same order as your mapping
                ent_order = [None] * len(self.idx_ent_dict)
                for iri, idx in self.idx_ent_dict.items():
                    ent_order[idx] = iri
                self.emb_entities = self.get_embeddings_from_csv(tmp_emb_folder + emb_typ + '/', 'all_entities_embeddings_final.csv', ent_order)
                print(f"  >> #mapped entities = {len(self.idx_ent_dict)}, #loaded embeddings = {len(self.emb_entities)}")

            # ── Relations ──────────────────────────────────────────────────────────
            rel_pkl = os.path.join(tmp_emb_folder, emb_typ, 'relation.pkl')
            if os.path.exists(rel_pkl):
                self.emb_relation = self.get_embeddings(tmp_emb_folder + emb_typ + '/', 'relation.pkl')
            else:
                rel_order = [None] * len(self.idx_rel_dict)
                for iri, idx in self.idx_rel_dict.items():
                    rel_order[idx] = iri
                self.emb_relation = self.get_embeddings_from_csv(tmp_emb_folder + emb_typ + '/', 'all_relations_embeddings_final.csv', rel_order )

            # ── Times ─────────────────────────────────────────────────────────────
            # if you have a .pkl for times, keep it; otherwise skip or load a CSV
            time_pkl = os.path.join(tmp_emb_folder, emb_typ, 'time.pkl')
            if os.path.exists(time_pkl):
                self.emb_time = self.get_embeddings(tmp_emb_folder + emb_typ + '/', 'time.pkl')
            else:
                self.emb_time = []

            # DEBUG: print out the exact counts
            #loaded_count = self.emb_entities.shape[0] if hasattr(self.emb_entities, "shape") else len(self.emb_entities)
            #mapped_count = len(self.idx_ent_dict)
            #print("Debug ENTITY COUNT")
            #print(f" -mapped entities (idx_ent_dict): {mapped_count})")
            #print(f"- loaded entities (idx_ent_dict): {loaded_count}")
            #print(f" -difference: {loaded_count-mapped_count}")
            # peek at the first few IDs in your map and the first few embeddings
            #print(" -sample idx_ent_dict keys:", list(self.idx_ent_dict.keys()) [:5])
            #if hasattr(self.emb_entities, "shape"):
            #    print(" -sample emb_entities rows:\n", self.emb_entities[:3])

            self.num_entities = len(self.emb_entities)
            assert len(self.emb_entities) == len(self.idx_ent_dict), "Entity count mismatch"
            self.num_relations = len(self.emb_relation)
            self.num_times = len(self.emb_time)
            self.idx_train_set = self.get_ids_dict(selected_dataset_data_dir+"train/train")
            self.idx_test_set = self.get_ids_dict(selected_dataset_data_dir+"test/test")
            self.idx_valid_set = self.get_ids_dict(selected_dataset_data_dir+"valid/valid")

            # -------------MY EDITITNG-------------------------
            # ——— DEBUG: check for out-of-bounds IDs in each split ———

            def check_oob(name, triplets, num_ent):
                # collect all subject & object indices
                idxs = [t[0] for t in triplets] + [t[2] for t in triplets]
                if not idxs:
                    return
                max_id = max(idxs)
                if max_id >= num_ent:
                    missing = sorted({i for i in idxs if i >= num_ent})
                    print(f"❌ OOB in {name}: max_id={max_id} >= num_entities={num_ent}")
                    print(f"    sample missing IDs: {missing[:10]}")
                    raise ValueError(f"Aborting: {name} contains invalid entity IDs.")

            # run checks
            check_oob("TRAIN", self.idx_train_set, self.num_entities)
            check_oob("TEST", self.idx_test_set, self.num_entities)
            check_oob("VALID", self.idx_valid_set, self.num_entities)
            # ————————————————————————————————————————————————
            # -------------MY EDITITNG-------------------------

    # Function to find a key by its value in a dictionary

    def load_id_split(self, file_path):
        out = []
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    out.append(list(map(int, parts)))
        return out

    def load_range_data(self, file_path):
        print(f"[DEBUG] Attempting to load range data from {file_path}")
        raw_lines = self.read_raw_lines(file_path)
        if not raw_lines:
            print(f"[WARNING] No data loaded from {file_path}")
            return []

        idx_data = []
        for line in raw_lines:
            try:
                parts = line.split('\t')  # tab-separated: h  r  t  y1  y2
                if len(parts) != 5:
                    print(f"[WARNING] Skipping malformed line (expected 5 columns): {line}")
                    continue

                h, r, t, y1, y2 = parts

                # strip to last segment (works for both <...> and plain URLs)
                def last_seg(x):
                    x = x.strip()
                    if x.startswith("<") and x.endswith(">"):
                        x = x[1:-1]
                    return x.rstrip("/").split("/")[-1]

                import re

                def norm_ent(x: str) -> str:
                    # works for <...>, http(s)://.../entity/Qxxx, dbpedia/.../resource/...
                    x = x.strip()
                    if x.startswith("<") and x.endswith(">"):
                        x = x[1:-1]
                    seg = x.rstrip("/").split("/")[-1]
                    # prefer bare Qxxxx at the end if present
                    m = re.search(r"(Q\d+)$", seg)
                    return m.group(1) if m else seg

                def norm_rel(x: str) -> str:
                    x = x.strip()
                    if x.startswith("<") and x.endswith(">"):
                        x = x[1:-1]
                    seg = x.rstrip("/").split("/")[-1]  # e.g., "Property:P729" or "P729"
                    # strip "Property:" if present
                    if seg.startswith("Property:"):
                        seg = seg.split(":", 1)[1]
                    # fallback: extract trailing Pxxxx if embedded
                    m = re.search(r"(P\d+)$", seg)
                    return m.group(1) if m else seg

                h_id = normalize_ent_id(h)
                r_id = normalize_rel_id(r)
                t_id = normalize_ent_id(t)

                h_idx = self.idx_ent_dict.get(h_id)
                r_idx = self.idx_rel_dict.get(r_id)
                t_idx = self.idx_ent_dict.get(t_id)

                y1_idx = self.idx_time_dict.get(str(int(float(y1))))
                y2_idx = self.idx_time_dict.get(str(int(float(y2))))

                if None in (h_idx, r_idx, t_idx, y1_idx, y2_idx):
                    print(f"[WARNING] Missing index for line: {line} -> "
                          f"(h={h_id}:{h_idx}, r={r_id}:{r_idx}, t={t_id}:{t_idx}, y1={y1}:{y1_idx}, y2={y2}:{y2_idx})")
                    continue

                idx_data.append([h_idx, r_idx, t_idx, y1_idx, y2_idx])

            except Exception as e:
                print(f"[ERROR] Failed to process line {line}: {e}")
                continue

        print(f"[DEBUG] Loaded {len(idx_data)} entries from {file_path}")

        # DEBUG coverage summary
        try:
            total = len(raw_lines)
            print(
                f"[DEBUG] Coverage for {file_path}: kept={len(idx_data)} / raw={total}  ({(len(idx_data) / max(1, total)) * 100:.2f}%)")
        except Exception:
            pass

        return idx_data

        ''' dataset = []
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) != 5:
                    print(f"[WARNING] Skipped malformed line: {line.strip()}")
                    continue
                s, p, o, y1, y2 = parts
                # Strip URIs to get entity/relation/time IDs
                s_id = s.rsplit("/", 1)[-1]
                p_id = p.rsplit("/", 1)[-1]
                o_id = o.rsplit("/", 1)[-1]

                try:
                    s_idx = self.idx_ent_dict[s_id]
                    p_idx = self.idx_rel_dict[p_id]
                    o_idx = self.idx_ent_dict[o_id]
                    y1_idx = self.idx_time_dict[str(int(float(y1)))]
                    y2_idx = self.idx_time_dict[str(int(float(y2)))]
                except KeyError:
                    continue

                dataset.append((s_idx, p_idx, o_idx, y1_idx, y2_idx))
        print(f" Loaded {len(dataset)} entries from {file_path}")
        return dataset '''

    def read_raw_lines(self, file_path):
        if not os.path.exists(file_path):
            print(f"[ERROR] File not found: {file_path}")
            return []
        with open(file_path, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
            print(f"[DEBUG] Read {len(lines)} raw lines from {file_path}")
            if lines:
                print(f"[DEBUG] First raw line: {lines[0]}")
            return lines

        """ raw_data = []
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) != 5:
                    continue
                s, p, o, y1, y2 = parts
                raw_data.append((s, p, o, y1, y2))
        return raw_data """

    def process_KGE_only_data(self, selected_dataset_data_dir, args, valid_ratio):
        self.idx_train_set = []
        self.idx_test_set = []
        self.idx_valid_set = []

        # reading train and test sets
        self.train_set = list(
            (self.load_data(selected_dataset_data_dir + "train/", data_type="train")))
        self.test_set = list(
            (self.load_data(selected_dataset_data_dir + "test/", data_type="test")))
        self.test_set, self.valid_set = self.generate_test_valid_set(self, self.test_set, valid_ratio)
        # negative triples generation
        if args.negative_triple_generation != "corrupted-time-based":
            self.train_set, count = self.generate_negative_triples(self.train_set, "corrupted-triple-based")
            self.test_set, count = self.generate_negative_triples(self.test_set, "corrupted-triple-based")
            self.valid_set, count = self.generate_negative_triples(self.valid_set, "corrupted-triple-based")

        self.idx_ent_dict = dict()
        self.idx_rel_dict = dict()

        self.data = self.train_set + self.test_set + self.valid_set
        self.entities = self.get_entities(self.data)
        self.relations = self.get_relations(self.data)
        # self.save_all_resources(self.entities, selected_dataset_data_dir, is_entity=True)
        # self.save_all_resources(self.relations, selected_dataset_data_dir, is_entity=False)
        # exit(1)
        # Generate integer mapping
        for i in self.entities:
            self.idx_ent_dict[i] = len(self.idx_ent_dict)
        for i in self.relations:
            self.idx_rel_dict[i] = len(self.idx_rel_dict)

        self.emb_entities = self.get_embeddings_from_csv(selected_dataset_data_dir + 'embeddings/'+args.emb_type,
                                                         '/entities_embeddings.csv', self.entities)
        self.emb_relation = self.get_embeddings_from_csv(selected_dataset_data_dir + 'embeddings/'+args.emb_type,
                                                         '/relations_embeddings.csv', self.relations)

        self.num_entities = len(self.emb_entities)
        assert len(self.emb_entities) == len(self.idx_ent_dict), "Entity count mismatch"
        self.num_relations = len(self.emb_relation)
        self.num_times = 0

        # creating ids of the train andtest sets
        i = 0
        for (s, p, o, label) in self.train_set:
            if self.idx_ent_dict.keys().__contains__(s) and self.idx_rel_dict.keys().__contains__(
                    p) and self.idx_ent_dict.keys().__contains__(o):
                idx_s, idx_p, idx_o, label = min(self.idx_ent_dict[s], self.num_entities - 1) , self.idx_rel_dict[p], self.idx_ent_dict[o], label
                if label == 'True' or label == 1:
                    label = 1
                else:
                    label = 0
                ver = i
                self.idx_train_set.append([int(idx_s), int(idx_p), int(idx_o), 0, ver, label])
            else:
                print("check:" + s + "," + o)
            i = i + 1
        i = 0
        for (s, p, o, label) in self.test_set:
            if self.idx_ent_dict.keys().__contains__(s) and self.idx_rel_dict.keys().__contains__(
                    p) and self.idx_ent_dict.keys().__contains__(o):
                idx_s, idx_p, idx_o, label = min(self.idx_ent_dict[s], self.num_entities - 1) , self.idx_rel_dict[p], self.idx_ent_dict[o], label
                if label == 'True' or label == 1:
                    label = 1
                else:
                    label = 0
                ver = i
                self.idx_test_set.append([int(idx_s), int(idx_p), int(idx_o), 0, ver, label])
            else:
                print("check:" + s + "," + o)
            i = i + 1
        i = 0
        for (s, p, o, label) in self.valid_set:
            if self.idx_ent_dict.keys().__contains__(s) and self.idx_rel_dict.keys().__contains__(
                    p) and self.idx_ent_dict.keys().__contains__(o):
                idx_s, idx_p, idx_o, label = min(self.idx_ent_dict[s], self.num_entities - 1) , self.idx_rel_dict[p], self.idx_ent_dict[o], label
                if label == 'True' or label == 1:
                    label = 1
                else:
                    label = 0
                ver = i
                self.idx_valid_set.append([int(idx_s), int(idx_p), int(idx_o), 0, ver, label])
            else:
                print("check:" + s + "," + o)
            i = i + 1
        print("loading train and test is done")

        # ——— STEP: generate paired indices for range‐prediction ———
        from utils_TP.dataset_classes import make_pairs

        # -------------MY EDITITNG-------------------------
        # ——— DEBUG: check for out-of-bounds IDs in each split ———

        def check_oob(name, triplets, num_ent):
            # collect all subject & object indices
            idxs = [t[0] for t in triplets] + [t[2] for t in triplets]
            if not idxs:
                return
            max_id = max(idxs)
            if max_id >= num_ent:
                missing = sorted({i for i in idxs if i >= num_ent})
                print(f"❌ OOB in {name}: max_id={max_id} >= num_entities={num_ent}")
                print(f"    sample missing IDs: {missing[:10]}")
                raise ValueError(f"Aborting: {name} contains invalid entity IDs.")

        # run checks
        check_oob("TRAIN", self.idx_train_set, self.num_entities)
        check_oob("TEST", self.idx_test_set, self.num_entities)
        check_oob("VALID", self.idx_valid_set, self.num_entities)
        # ————————————————————————————————————————————————
        # -------------MY EDITITNG-------------------------


    def get_key(self,dictionary, value):
        for key, val in dictionary.items():
            if val == value:
                return key
        return None
    def generate_negative_triples(self, data, type="time-based"):

        data2 = []
        data_final = []
        count_positve = 0
        i =0
        if type=="corrupted-time-based":
            times = []
            for (s, p, o, time, label) in data:
                if label == 'True' or label == 1:
                    times.append(time)
                    data2.append([s, p, o, time, 1])
                i = i + 1
            count_positve = len(data2)
            random.shuffle(times)
            data3 = []
            for j in range(len(data2)):
                item = data2.__getitem__(j)
                tim = times.__getitem__(j)
                data3.append([item[0],item[1],item[2],tim,0])

            data_final = data2 + data3
        else:
            relations =  []
            i = 0
            for (s, p, o, label) in data:
                if label == 'True' or label == 1:
                    relations.append(p)
                    data2.append([s, p, o, 1])
                i = i + 1
            relations = set(relations)
            idx_relations = dict()
            for rel in relations:
                idx_relations[rel] = len(idx_relations)

            data3 = []
            count_positve = len(data2)
            for j in range(len(data2)):
                item = data2.__getitem__(j)
                rr =  idx_relations[item[1]]
                new_idx = 0
                if rr < len(idx_relations)-1:
                    new_idx = rr+1
                new_r = self.get_key(idx_relations, new_idx)
                data3.append([item[2], new_r, item[0], 0])

            data_final = data2 + data3
        # data_final.append(data3)
        return data_final, count_positve

    def generate_only_true_triples(self, data):
        data2 = []
        data_final = []
        i =0
        times = []
        for (s, p, o, time, label) in data:
            if label == 'True\n':
                label = label[:-1]
            if label == 'True':
                times.append(time)
                data_final.append([s, p, o, time, label])
            if label == 1:
                times.append(time)
                data_final.append([s, p, o, time, label])

            i = i + 1

        # data_final.append(data3)
        return data_final

    @staticmethod
    def get_veracity_data(self, train_emb):
        embeddings_train = dict()
        i = 0
        for train in train_emb:
            embeddings_train[i] = float(str(train[3]).replace(".\n",""))
            i += 1

        return pd.DataFrame(embeddings_train.values())

    @staticmethod
    def update_and_match_triples_start(self, selected_dataset_data_dir, type, file_name, data_set1, data_set2,  properties_split = None, veracity = False):
        if veracity==False:
            if (os.path.exists(selected_dataset_data_dir + type+ "/"+ file_name)):
                self.set_time_final = list(self.load_data(selected_dataset_data_dir + type+"/", data_type=str(file_name).replace(".txt",""),pred=True))
            else:
                if len(data_set1) != len(data_set2):
                    self.set_time_final = self.update_match_triples(data_set1, data_set2)
                else:
                    self.set_time_final = data_set2
                self.save_triples(selected_dataset_data_dir, type+"/"+file_name, self.set_time_final)
        else:
            tt = "properties/train/" if (file_name.__contains__("train")) else "properties/test/"
            split = "" if (properties_split==None) else tt+"correct/" +properties_split + "_"
            if (os.path.exists(selected_dataset_data_dir + type+ "/"+ split+ file_name)):
                self.set_time_final = list(self.load_data(selected_dataset_data_dir + type+"/"+split , data_type=str(file_name).replace(".txt",""),pred=True))
            else:
                self.set_time_final = self.update_match_triples(data_set1, data_set2, veracity=veracity)
                self.save_triples(selected_dataset_data_dir, type+"/"+split+file_name, self.set_time_final,veracity=veracity)
        return self.set_time_final

    def is_valid_test_available(self):
        if len(self.idx_valid_set) > 0 and len(self.idx_test_set) > 0:
            return True
        return False

    # @staticmethod
    # def load_triples(data_dir, type, triples):
    #     with open(data_dir + type + '.txt', "r") as f:
    #         for item in triples:
    #             f.write("%s\n" % item)
    @staticmethod
    def save_triples(data_dir,type, triples,veracity=False):
        if veracity==False:
            with open(data_dir + type, "w") as f:
                for item in triples:
                    f.write(""+(item[0])+"\t"+(item[1])+"\t"+(item[2])+"\t"+str(item[3])+"\t"+str(item[4])+"\n")
        else:
            with open(data_dir + type, "w") as f:
                for item in triples:
                    f.write(""+str(item[0])+"\t"+str(item[1])+"\t"+str(item[2])+"\t"+str(item[3])+"\n")
    @staticmethod
    def save_all_resources(list_all_entities, data_dir, sub_path="", is_entity=True):
        if is_entity:
            with open(data_dir+sub_path+'all_entities.txt',"w") as f:
                for item in list_all_entities:
                    f.write("%s\n" % item)
        else:
            with open(data_dir + sub_path + 'all_relations.txt', "w") as f:
                for item in list_all_entities:
                    f.write("%s\n" % item)

    @staticmethod
    def generate_test_valid_set(self, test_set, valid_ratio):
        test_data = []
        valid_data = []
        i = 0
        sent_i = 0
        for data in test_set:
            if i % valid_ratio == 0:
                valid_data.append(data)
            else:
                test_data.append(data)

            i += 1
        return  test_data, valid_data
    @staticmethod
    def generate_test_valid_sentence_set(self, test_set, valid_ratio):
        valid_indices = list(range(0, len(test_set), valid_ratio))  # Indices for validation set
        all_indices = list(range(len(test_set)))

        test_indices = [idx for idx in all_indices if idx not in valid_indices]  # Indices for test set

        test_data = test_set.iloc[test_indices]  # Extract test set
        valid_data = test_set.iloc[valid_indices]  # Extract validation set

        return test_data, valid_data

    def get_ids_dict(self, dict_file_path):
        """
        Parse mapping files and return a dict: {key: int_index}.

        Supports:
          - entities / relations:
              ID <TAB> KEY
              KEY <TAB> ID

          - times:
              ID <TAB> YEAR
              YEAR <TAB> ID

        For times we use a robust full-column check instead of the old heuristic.
        """
        import os
        import re

        pairs = []
        with open(dict_file_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) != 2:
                    continue
                a, b = parts
                pairs.append((a, b))

        def _is_int(tok: str) -> bool:
            return re.fullmatch(r"[+-]?\d+", tok) is not None

        def _is_iri_token(tok: str) -> bool:
            if "http" in tok or "entity/" in tok or "Property:" in tok:
                return True
            if re.search(r"(Q\d+|P\d+)$", tok):
                return True
            return not _is_int(tok)

        # -------- SPECIAL CASE FOR TIME MAPS --------
        fname = os.path.basename(dict_file_path).lower()
        if "time" in fname or fname in {"times", "times_map.tsv", "start_times", "end_times"}:
            numeric_pairs = [(a, b) for a, b in pairs if _is_int(a) and _is_int(b)]
            if len(numeric_pairs) == len(pairs) and len(pairs) > 0:
                col0 = [int(a) for a, b in numeric_pairs]
                col1 = [int(b) for a, b in numeric_pairs]
                n = len(numeric_pairs)

                def looks_like_index(vals):
                    s = set(vals)
                    return (
                            len(s) == n and
                            min(vals) >= 0 and
                            0 in s and
                            max(vals) <= max(n * 2, 10)
                    )

                ids_dict = {}

                # case 1: first column is id, second is year
                if looks_like_index(col0):
                    for a, b in numeric_pairs:
                        ids_dict[str(int(b))] = int(a)
                    return ids_dict

                # case 2: second column is id, first is year
                if looks_like_index(col1):
                    for a, b in numeric_pairs:
                        ids_dict[str(int(a))] = int(b)
                    return ids_dict

                raise ValueError(
                    f"Could not determine orientation for time map: {dict_file_path}"
                )

        # -------- GENERAL CASE FOR ENTITIES / RELATIONS --------
        sample = pairs[:50] if len(pairs) > 50 else pairs

        a_looks_idx = 0
        for a, b in sample:
            if _is_int(a) and _is_iri_token(b):
                a_looks_idx += 1

        b_looks_idx = 0
        for a, b in sample:
            if _is_int(b) and _is_iri_token(a):
                b_looks_idx += 1

        if b_looks_idx > a_looks_idx:
            orientation = "KEY_TAB_ID"
        else:
            orientation = "ID_TAB_KEY"

        ids_dict = {}
        if orientation == "ID_TAB_KEY":
            for a, b in pairs:
                if not _is_int(a):
                    continue
                ids_dict[b] = int(a)
        else:
            for a, b in pairs:
                if not _is_int(b):
                    continue
                ids_dict[a] = int(b)

        widened = {}
        for k, v in list(ids_dict.items()):
            kk = k.strip("<>")
            seg = kk.rstrip("/").split("/")[-1]
            widened[seg] = v
            if seg.startswith("Property:"):
                widened[seg.split(":", 1)[1]] = v
        ids_dict.update(widened)

        return ids_dict

    #    ids_dict = dict()
    #    data = []
    #    with open("%s" % (dict_file_path), "r") as f:
    #        for datapoint in f:
                #datapoint = datapoint.split()
    #            parts = datapoint.strip().split()
    #            if len(parts) == 2:
    #                ids_dict[parts[1]] = int(parts[0])
                #elif len(datapoint)==5:
                 #   arr = []
                  #  for tt in datapoint:
                   #     arr.append(int(tt))
                    #arr.append(True)
                    #data.append(arr)
    #            elif len(parts) == 5:
    #                s_iri, p_iri, o_iri, t_str, lbl_str = parts
                    # look up numeric IDs
    #                idx_s = self.idx_ent_dict[s_iri]
    #                idx_p = self.idx_rel_dict[p_iri]
    #                idx_o = self.idx_ent_dict[o_iri]
    #                idx_t = self.idx_time_dict[t_str]
                    # parse label ("True." or "False.")
    #                lbl = 1 if lbl_str.lower().startswith("true") else 0
    #                data.append([idx_s, idx_p, idx_o, idx_t, lbl])
    #            else:
    #                print("invalid format")
    #                exit(1)
    #    if len(ids_dict) > 0:
    #        return ids_dict
    #    else:
    #        return data
    @staticmethod
    def load_data(data_dir, data_type, pred=False):
        try:
            data = []
            if pred == False:
                with open("%s%s" % (data_dir, data_type), "r") as f:
                    for datapoint in f:
                        datapoint = datapoint.split("\t")
                        if len(datapoint) == 4:
                            s, p, o, label = datapoint
                            if label == '.\n': # TODO if false triples are also provided then label could be false or??
                                label = 1
                            elif label == 'True' or label == 1 or label == '1':
                                label = 1
                            else:
                                label = 0
                            data.append((s, p, o, label))
                        elif len(datapoint) == 3:
                            s, p, label = datapoint
                            assert label == 'True' or label == 'False'
                            if label == 'True':
                                label = 1
                            else:
                                label = 0
                            data.append((s, p, 'DUMMY', label))
                        elif len(datapoint) == 5:
                            if datapoint[4]==".\n":
                                s, p, o, label, dummy = datapoint
                                label = label.replace("\n", "")
                                if label == 'True' or label == '1.0' or label == 1.0 or label == '1' or label == 1:
                                    label = 1
                                else:
                                    label = 0
                                data.append((s, p, o,"N/A", label))
                            else:
                                s, p, o, time, label = datapoint
                                label=label.replace("\n","")
                                assert label == 'True.' or label == 'False.'         #main code
                                if label == 'True' or label == '1' or label == 1:
                                    label = 1
                                else:
                                    label = 0
                                data.append((s, p, o, time, label))
                        else:
                            raise ValueError
            else:
                with open("%s%s" % (data_dir, data_type), "r") as f:
                    for datapoint in f:
                        datapoint = datapoint.split('\t')
                        if len(datapoint) == 4:
                            s, p, o, label = datapoint
                            label = str(label).replace(".\n", "")
                            label = str(label).replace("\n","")
                            data.append((s, p, o, float(label)))
                        elif len(datapoint) == 3:
                            s, p, label = datapoint
                            label = str(label).replace("\n", "")
                            data.append((s, p, 'DUMMY', float(label)))
                        elif len(datapoint) == 5:
                            s, p, o, time, label = datapoint
                            label = str(label).replace("\n", "")
                            if label=="." and str(time).__contains__("^<http://www.w3.org/2001/XMLSchema#double>"):
                                label = time
                                label = str(label).replace("\"^^<http://www.w3.org/2001/XMLSchema#double>", "")
                                data.append((s, p, o, float(label)))
                            elif label=="." and (str(time).startswith('0.') or time == '1.0'):
                                data.append((s, p, o, float(time)))
                            else:
                                data.append((s, p, o, time, float(label)))
                        else:
                            raise ValueError
        except FileNotFoundError as e:
            print(e)
            print('Add empty.')
            data = []
        return data
    @staticmethod
    def get_mapped_entities(data_dir, file_name):
        mapping_entities = dict()
        with open("%s%s.txt" % (data_dir, file_name), "r") as f:
            for datapoint in f:
                datapoint = datapoint.split("	->	")
                if len(datapoint) == 2:
                    mapping_entities[datapoint[0]] = datapoint[1].replace("\n","")
        return mapping_entities

    def check_if_not_equal_size(self, data_set1, data_set2):
        data = []
        if len(data_set1)!=len(data_set2):
            train_set_time1 = deepcopy(data_set1)
            for tp in data_set2:
                found = False
                for tpt in train_set_time1:
                    if (tp[0] == tpt[0] and tp[1] == tpt[1].replace("Of", "") and tp[2] == tpt[2]):
                        data.append([tpt[0], tpt[1], tpt[2], tpt[3]])
                        found = True
                        break
                    elif (tp[2] == tpt[0] and tp[0] == tpt[2]):
                        data.append([tpt[0], tpt[1], tpt[2], tpt[3]])
                        found = True
                        break
                if found == False:
                    print("Embeddings not found: excluded triple:" + str(tp))
                else:
                    train_set_time1.remove(tpt)

            return data
        else:
            return data_set1

    # update date set 1 and match with dataset 2
    @staticmethod
    def update_match_triples(data_set1, data_set2, veracity=False, final= False):
        data = []
        data_set21 = deepcopy(data_set2)
        # subs = [tp2[0] for tp2 in data_set2]
        # preds = [tp2[1].replace("Of","") for tp2 in data_set2]
        # objs = [tp2[2] for tp2 in data_set2]
        for tp in data_set1:
            found = False
            if veracity == False:
                for tpt in data_set21:
                    # if tpt[0].__contains__('Amadou_Toumani') and (tp[2].__contains__('Amadou_Toumani_')):
                    #     print("test")
                    if (tp[0] == tpt[0] and tp[1] == tpt[1].replace("Of","") and tp[2] == tpt[2]):
                        data.append([tp[0],tp[1],tp[2],tpt[3],tpt[4]])
                        found = True
                        break
                    elif(tp[2] == tpt[0] and tp[0] == tpt[2]):# to cover negative triples we are doing like this, swaping the sub and obj and not checking the predicate
                        data.append([tpt[0], tpt[1], tpt[2], tpt[3], 'False'])
                        found = True
                        break

                if found == False:
                    print("not found:"+ str(tp))
            elif veracity==True and final==True: # final is for second check
                for tpt in data_set21:
                    if (tp[0] == tpt[0] and tp[1] == tpt[1].replace("Of", "") and tp[2] == tpt[2]):
                        data.append([tp[0], tp[1], tp[2], tp[3], tp[4]])
                        found = True
                        break
                    elif (tp[2] == tpt[0] and tp[1] == tpt[1].replace("Of", "") and tp[0] == tpt[2]):# to cover negative triples we are doing like this, swaping the sub and obj and not checking the predicate
                        data.append([tp[0], tp[1], tp[2], tp[3], tp[4]])
                        found = True
                        break
                if found == False:
                    print("not found:"+ str(tp))
                else:
                    data_set21.remove(tpt)

            else:
                for tpt in data_set21:
                    if (tp[0] == tpt[0] and tp[1] == tpt[1].replace("Of", "") and tp[2] == tpt[2]):
                        data.append([tp[0], tp[1], tp[2], tp[3]])
                        found = True
                        break
                    elif (tp[2] == tpt[0] and tp[0] == tpt[2]): # to cover negative triples we are doing like this, swaping the sub and obj and not checking the predicate
                        data.append([tp[0], tp[1], tp[2], tp[3]])
                        found = True
                        break
                if found == False:
                    print("not found:"+ str(tp))
                # break

                # else:
                #     print("problematic triple:"+ str(tp))

        return data

    @staticmethod
    def load_data_with_time(data_dir, data_type, mapped_entities=None, prop = None):
        try:
            data = []
            with open("%s%s.txt" % (data_dir, data_type), "r") as f:
                for datapoint in f:
                    datapoint = datapoint.split("\t")
                    if len(datapoint) >= 5:
                        if len(datapoint) >5:
                            datapoint[5] = '_'.join(datapoint[4:])
                        s, p, o, time, loc = datapoint[0:5]
                        if prop!=None:
                            if not str(p).__eq__(prop+"Of"):
                                continue
                        s = "http://dbpedia.org/resource/" + s
                        if (mapped_entities!=None and s in mapped_entities.keys()):
                            s = mapped_entities[s]
                        p = "http://dbpedia.org/ontology/" + p
                        o = "http://dbpedia.org/resource/" + o
                        if (mapped_entities!=None and o in mapped_entities.keys()):
                            o = mapped_entities[o]
                        data.append(("<" + s + ">", "<" + p + ">", "<" + o + ">", time, "True"))
                    elif len(datapoint) == 3:
                        s, p, label = datapoint
                        assert label == 'True' or label == 'False'
                        if label == 'True':
                            label = 1
                        else:
                            label = 0
                        data.append((s, p, 'DUMMY', label))
                    else:
                        raise ValueError
        except FileNotFoundError as e:
            print(e)
            print('Add empty.')
            data = []
        return data
    @staticmethod
    def get_relations(data):
        relations = sorted(list(set([d[1] for d in data])))
        return relations

    @staticmethod
    def get_entities(data):
        entities = sorted(list(set([d[0] for d in data] + [d[2] for d in data])))
        return entities

    @staticmethod
    def get_times(data):
        times = set()
        for d in data:
            if len(d) >= 5:
                for y in (d[3], d[4]):
                    try:
                        times.add(str(int(float(y))))
                    except Exception:
                        pass
            else:
                times.add(str(int(float(d[3]))))
        #times = sorted(list(set([d[3] for d in data])))
        return sorted(times)
    # / home / umair / Documents / pythonProjects / HybridFactChecking / Embeddings / ConEx_dbpedia
    @staticmethod
    def get_embeddings_from_csv(path,name, order):

        embd = pd.read_csv("%s%s" % (path, name))

        first_col = embd.columns[0]

        embd['key'] = pd.Categorical(embd[first_col], categories=order, ordered=True)

        # Sort the DataFrame based on the 'Fruits' column
        sorted_df = embd.sort_values(by="key")

        # --- trim any “reserved” embeddings that aren’t in our mapping file ---
        extra = sorted_df.shape[0] - len(order)
        if extra > 0:
            print(f" Trimming off {extra} reserved-token embeddings from CSV")
            sorted_df = sorted_df.iloc[extra:].reset_index(drop=True)

        #return sorted_df.iloc[:, 1:]
        return sorted_df.drop([first_col, 'key'], axis=1).reset_index(drop=True)

    @staticmethod
    def get_embeddings(path, name):
        import os
        full = f"{path}{name}"
        if name.endswith(".npy"):
            import numpy as np
            arr = np.load(full)  # shape [N, d]
            return arr  # keep as numpy; callers handle conversion / alignment
        if name.endswith(".pkl"):
            import inspect, torch
            load_kwargs = {"map_location": torch.device("cpu")}
            if "weights_only" in inspect.signature(torch.load).parameters:
                try:
                    load_kwargs["weights_only"] = True
                    emb_obj = torch.load(full, **load_kwargs)
                except Exception:
                    load_kwargs.pop("weights_only", None)
                    emb_obj = torch.load(full, **load_kwargs)
            else:
                emb_obj = torch.load(full, **load_kwargs)
            if isinstance(emb_obj, torch.nn.Embedding):
                return emb_obj.weight
            if isinstance(emb_obj, torch.nn.Parameter):
                return emb_obj.data
            return emb_obj  # tensor/np/list
        if name.endswith(".csv"):
            import pandas as pd
            embd = pd.read_csv(full, sep=",")
            last_column_name = embd.columns[-1]
            if str(embd[last_column_name]).__contains__("]"):
                embd[last_column_name] = embd[last_column_name].str.replace(']', '', regex=False)
            return embd.iloc[:, 1:]
        print("invalid embeddings format. Please use .npy, .csv or .pkl format")
        raise ValueError

    @staticmethod
    def get_comma_seperated_embeddings(idxs, path, name):
        embeddings = dict()
        # print("%s%s.txt" % (path,name))
        with open("%s%s.txt" % (path, name), "r") as f:
            for datapoint in f:
                data = datapoint.split('> ,')
                if datapoint.startswith("<http://dbpedia.org/resource/Abu_Jihad_("):
                    print(datapoint)
                if len(data) == 1:
                    data = datapoint.split('>\",')
                if len(data) > 1:
                    data2 = data[0] + ">", data[1].split(',')
                    # test = data2[0].replace("\"","").replace("_com",".com").replace("Will-i-am","Will.i.am").replace("Will_i_am","Will.i.am")
                    test = data2[0].replace("\"", "")
                    if test in idxs:
                        embeddings[test] = data2[1]
                    # else:
                    #     print('Not in embeddings:',datapoint)
                    # exit(1)
                # else:
                #     print('Not in embeddings:',datapoint)
                #     exit(1)
        for emb in idxs:
            if emb not in embeddings.keys():
                print("this is missing in embeddings file:" + emb)
                exit(1)

        if len(idxs) > len(embeddings):
            print("embeddings missing")
            exit(1)
        embeddings_final = dict()
        for emb in idxs.keys():
            if emb in embeddings.keys():
                embeddings_final[emb] = embeddings[emb]
            else:
                print('no embedding', emb)
                exit(1)

        return embeddings_final.values()
    @staticmethod
    def get_copaal_veracity(path, name, train_set):
        emb = dict()

        embeddings_train = dict()
        # print("%s%s" % (path,name))

        i = 0
        train_i = 0
        found = False
        with open("%s%s" % (path, name), "r") as f:
            for datapoint in f:
                if datapoint.startswith("0,1,2"):
                    continue
                else:
                    emb[i] = datapoint.split(',')
                    try:
                        for dd in train_set:
                            # figure out some way to handle this first argument well
                            if (emb[i][0] == dd[0].replace(',', '')) and (emb[i][1] == dd[1].replace(',', '')) and (
                                    emb[i][2] == dd[2].replace(',', '')):
                                # print('train data found')
                                embeddings_train[train_i] =np.append(emb[i][:3],emb[i][-1].replace("\n",""))
                                train_i += 1
                                found = True
                                break

                            # else:
                            #     print('error')
                            # exit(1)
                    except:
                        print('ecception')
                        exit(1)
                    if found == False:
                        if (train_i >= len(train_set)):
                            break
                        else:
                            print("some training data missing....not found:" + str(emb[i]))
                            exit(1)
                    i = i + 1
                    found = False

                    # i = i+1
            embeddings_train_final = dict()
            jj = 0
            # print("sorting")
            for embb in train_set:
                ff = False
                for embb2 in embeddings_train.values():
                    if ((embb[0].replace(',', '') == embb2[0].replace(',', '')) and (
                            embb[1].replace(',', '') == embb2[1].replace(',', '')) and (
                            embb[2].replace(',', '') == embb2[2].replace(',', ''))):
                        embeddings_train_final[jj] = embb2
                        jj = jj + 1
                        ff = True
                        break
                if ff == False:
                    print("problem: not found")
                    exit(1)

        if len(train_set) != len(embeddings_train_final):
            print("problem")
            exit(1)
        return embeddings_train_final.values()
    @staticmethod
    def update_entity(self, ent):
        ent = ent.replace("+", "")
        if (ent.__contains__("&") or ent.__contains__("%")) and (
                (not ent.__contains__("%3F")) and (not ent.__contains__("%22"))):
            sub2 = ""

            for chr in ent:
                if chr == "&" or chr == "%":
                    break
                else:
                    sub2 += chr
            if ent[0]=="<":
                ent = sub2 + ">"
            else:
                ent = sub2

        if ent.__contains__("?"):
            ent = ent.replace("?", "%3F")

        if ent.__contains__("\"\""):
            ent= ent.replace("\"\"", "%22")
        if ent[0] == "\"" and ent[-1] == "\"":
            ent = ent[1:-1]
        if ent[0] == "\'" and ent[-1] == "\'":
            ent = ent[1:-1]
        return ent

    def without(self,d, key):
        new_d = d.copy()
        new_d2 = dict()
        new_d.pop(key)
        count = 0
        for dd in new_d.values():
            new_d2[count]=dd
            count+=1
        return new_d2
    @staticmethod
    def get_sent_embeddings(self, path, name, train_set):
        emb = dict()

        embeddings_train = dict()
        # print("%s%s" % (path,name))
        train_set_copy = deepcopy(train_set)
        i = 0
        train_i = 0
        found = False
        with open("%s%s" % (path, name), "r") as f:
            for datapoint in f:
                if datapoint.startswith("0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20"):
                    continue
                else:
                    if datapoint.startswith("http://dbpedia.org/resource/Vlado_Brankovic"):
                        print("test")
                    emb[i] = datapoint.split('\t')
                    try:
                        if emb[i][0] != "0":
                            for dd in train_set_copy:
                                # updated because factcheck results does not contained punctuations
                                sub = self.update_entity(self, dd[0])
                                pred = self.update_entity(self, dd[1])
                                obj = self.update_entity(self, dd[2])

                                emb[i][0] = self.update_entity(self, emb[i][0])
                                emb[i][1] = self.update_entity(self, emb[i][1])
                                emb[i][2] = self.update_entity(self, emb[i][2])

                                if ((emb[i][0] == sub) and
                                        (emb[i][1] == pred) and
                                        (emb[i][2] == obj)
                                        or
                                        ('<'+emb[i][0].lower()+'>' == sub.lower()) and
                                        ('<'+emb[i][1].lower()+'>' == pred.lower()) and
                                        ('<'+emb[i][2].lower()+'>' == obj.lower())):
                                    # print('train data found')
                                    emb[i][-1] = emb[i][-1].replace("'", "").replace("\n","")
                                    if (len(emb[i])) == ((768 * 3) +3 + 1):
                                        # because defacto scores are also appended at the end
                                        embeddings_train[train_i] = emb[i][:-1]
                                    elif (len(emb[i])) == ((768 * 3) +3):
                                        # emb[i][-1] = emb[i]
                                        embeddings_train[train_i] = emb[i]
                                    else:
                                        print("there is something fishy:"+str(emb[i]))
                                        exit(1)

                                    train_i += 1
                                    found = True
                                    break

                                # else:
                                #     print('error')
                                # exit(1)
                    except:
                        print('ecception')
                        exit(1)
                    if found==True:
                        train_set_copy.remove(dd)
                    # else:
                    #     train_set.remove(dd)
                    if found == False:
                        if (train_i >= len(train_set)):
                            break
                        else:
                            print("some training data missing....not found:" + str(emb[i]))
                            print(i)
                            print("test")
                            # exit(1)
                    i = i + 1
                    found = False

                    # i = i+1

        if len(train_set) != len(embeddings_train):
            print("problem: length of train and sentence embeddings arrays are different:train:"+str(len(train_set))+",emb:"+str(len(embeddings_train)))
            # exit(1)
        # following code is just for ordering the data in sentence vectors
        train_i = 0
        train_set_copy = deepcopy(train_set)
        embeddings_train_final = dict()
        for dd in train_set:
            found_data = False
            jj = 0
            for sd in embeddings_train.values():
                sub = self.update_entity(self, dd[0])
                pred =self.update_entity(self, dd[1])
                obj = self.update_entity(self, dd[2])
                sub1 = '<'+self.update_entity(self, sd[0])+'>'
                pred1 = '<'+self.update_entity(self, sd[1])+'>'
                obj1 = '<'+self.update_entity(self, sd[2])+'>'
                if (((sub == sub1) and (pred == pred1) and (obj == obj1))
                    or
                    (( sub1.lower()== sub.lower()) and ( pred1.lower() == pred.lower()) and
                     ( obj1.lower() == obj.lower()))):
                    embeddings_train_final[train_i] = sd
                    train_i+=1
                    found_data = True
                    break
                jj += 1
            if found_data== False:
                train_set_copy.remove(dd)
                print("missing train data from sentence embeddings file:"+str(dd))
            else:
                # print("to delete from list: "+str(sd))
                embeddings_train =  self.without(embeddings_train,jj)
                # embeddings_train = embeddings_train.dropna().reset_index(drop=True)
                # del embeddings_train[sd]

        train_set = deepcopy(train_set_copy)
        return embeddings_train_final.values(), train_set


    @staticmethod
    def update_copaal_veracity_score(self, train_emb):
        embeddings_train = dict()
        i = 0
        for train in train_emb:
            embeddings_train[i] = train[3:]
            i += 1

        return embeddings_train.values()

    @staticmethod
    def update_veracity_train_set(self, train_emb):
        embeddings_train = dict()
        i = 0
        for train in train_emb:
            embeddings_train[i] = train[3:]
            i += 1

        return embeddings_train.values()
    @staticmethod
    def update_sent_train_embeddings(self, train_emb):
        embeddings_train = dict()
        i=0
        for train in train_emb:
            embeddings_train[i] = train[3:]
            i+=1

        return embeddings_train.values()

    @staticmethod
    def get_veracity_test_valid_set(path, name, test_set, valid_set):
        embeddings_test, embeddings_valid = dict(), dict()
        emb = dict()
        # print("%s%s" % (path, name))
        found = False
        i = 0
        test_i = 0
        valid_i = 0
        with open("%s%s" % (path, name), "r") as f:
            for datapoint in f:
                if datapoint.startswith("0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20"):
                    continue
                else:
                    emb[i] = datapoint.split(',')
                    try:
                        for dd in test_set:
                            # figure out some way to handle this first argument well
                            if (emb[i][0].replace(',', '') == dd[0].replace(',', '')) and (
                                    emb[i][1].replace(',', '') == dd[1].replace(',', '')) and (
                                    emb[i][2].replace(',', '') == dd[2].replace(',', '')):
                                # print('test data found')
                                embeddings_test[test_i] = np.append(emb[i][:3],emb[i][-1].replace("\n",""))
                                test_i += 1
                                found = True
                                break
                        for vd in valid_set:
                            # figure out some way to handle this first argument well
                            if (emb[i][0].replace(',', '') == vd[0].replace(',', '')) and (
                                    emb[i][1].replace(',', '') == vd[1].replace(',', '')) and (
                                    emb[i][2].replace(',', '') == vd[2].replace(',', '')):
                                # print('valid data found')
                                embeddings_valid[valid_i] = np.append(emb[i][:3],emb[i][-1].replace("\n",""))
                                valid_i += 1
                                found = True
                                break
                        if found == False:
                            print("some data missing from test and validation sets..error" + str(emb[i]))
                            exit(1)
                        else:
                            found = False

                    except:
                        print('ecception')
                        exit(1)
                    i = i + 1

        embeddings_test_final, embeddings_valid_final = dict(), dict()
        i = 0
        for dd in test_set:
            for et in embeddings_test.values():
                if (et[0].replace(',', '') == dd[0].replace(',', '')) and (
                        et[1].replace(',', '') == dd[1].replace(',', '')) and (
                        et[2].replace(',', '') == dd[2].replace(',', '')):
                    embeddings_test_final[i] = et
                    i = i + 1
                    break
        i = 0
        for dd in valid_set:
            # print(dd)
            for et in embeddings_valid.values():
                if (et[0].replace(',', '') == dd[0].replace(',', '')) and (
                        et[1].replace(',', '') == dd[1].replace(',', '')) and (
                        et[2].replace(',', '') == dd[2].replace(',', '')):
                    embeddings_valid_final[i] = et
                    i = i + 1
                    break
        if (len(embeddings_valid_final) != len(valid_set)) and (len(embeddings_test_final) != len(test_set)):
            exit(1)
        return embeddings_test_final.values(), embeddings_valid_final.values()


    @staticmethod
    def get_sent_test_valid_embeddings(self, path, name, test_set, valid_set):
        embeddings_test, embeddings_valid = dict(),dict()
        emb = dict()
        # print("%s%s" % (path, name))
        found = False
        i = 0
        test_i = 0
        valid_i = 0
        with open("%s%s" % (path, name), "r") as f:
            for datapoint in f:
                if datapoint.startswith("0\t1\t2"):
                    continue
                else:
                    emb[i] = datapoint.split('\t')
                    try:
                        if emb[i][0] != "\"0\"":
                            for dd in test_set:
                                # figure out some way to handle this first argument well
                                sub = self.update_entity(self, dd[0])
                                pred = self.update_entity(self, dd[1])
                                obj = self.update_entity(self, dd[2])

                                emb[i][0] = self.update_entity(self, emb[i][0])
                                emb[i][1] = self.update_entity(self, emb[i][1])
                                emb[i][2] = self.update_entity(self, emb[i][2])


                                if  (((emb[i][0].replace(',', '') == sub.replace(',','')) and
                                        (emb[i][1].replace(',', '') == pred.replace(',','')) and (
                                        emb[i][2].replace(',', '') == obj.replace(',','')))
                                        or
                                        (('<'+emb[i][0].lower()+'>' == sub.lower()) and
                                        ('<'+emb[i][1].lower()+'>' == pred.lower()) and
                                        ('<'+emb[i][2].lower()+'>' == obj.lower()))):
                                    # print('test data found')
                                    emb[i][-1] = emb[i][-1].replace("'", "").replace("\n", "")
                                    if (len(emb[i])) == ((768 * 3) + 3 + 1):
                                        # because defacto scores are also appended at the end
                                        embeddings_test[test_i] = emb[i][:-1]
                                    elif (len(emb[i])) == ((768 * 3) + 3):
                                        # emb[i][-1] = emb[i]
                                        embeddings_test[test_i] = emb[i]
                                    else:
                                        print("there is something fishy:" + str(emb[i]))
                                        exit(1)
                                    # embeddings_test[test_i] = emb[i]
                                    test_i += 1
                                    found = True
                                    break
                            if found == False:
                                for vd in valid_set:
                                    sub = self.update_entity(self, vd[0])
                                    pred = self.update_entity(self, vd[1])
                                    obj = self.update_entity(self, vd[2])

                                    emb[i][0] = self.update_entity(self, emb[i][0])
                                    emb[i][1] = self.update_entity(self, emb[i][1])
                                    emb[i][2] = self.update_entity(self, emb[i][2])

                                    # figure out some way to handle this first argument well
                                    if (((emb[i][0].replace(',', '') == sub.replace(',', '')) and (
                                            emb[i][1].replace(',', '') == pred.replace(',', '')) and (
                                            emb[i][2].replace(',', '') == obj.replace(',', '')))
                                            or
                                            (('<' + emb[i][0].lower() + '>' == sub.lower()) and
                                             ('<' + emb[i][1].lower() + '>' == pred.lower()) and
                                             ('<' + emb[i][2].lower() + '>' == obj.lower()))):
                                        # print('valid data found')
                                        emb[i][-1] = emb[i][-1].replace("'", "").replace("\n", "")
                                        if (len(emb[i])) == ((768 * 3) + 3 + 1):
                                            # because defacto scores are also appended at the end
                                            embeddings_valid[valid_i] = emb[i][:-1]
                                        elif (len(emb[i])) == ((768 * 3) + 3):
                                            # emb[i][-1] = emb[i]
                                            embeddings_valid[valid_i] = emb[i]
                                        else:
                                            print("there is something fishy:" + str(emb[i]))
                                            exit(1)
                                        # embeddings_valid[valid_i] = emb[i]
                                        valid_i += 1
                                        found = True
                                        break
                            if found == False:
                                print("some data missing from test and validation sets..error"+ str(emb[i]))
                                    # exit(1)
                            else:

                                found = False

                    except:
                        print('ecception')
                        exit(1)
                    i = i + 1

        # embeddings_test_final, embeddings_valid_final = dict(), dict()
        # i = 0
        # for dd in test_set:
        #     for et in embeddings_test.values():
        #         if ((et[0].replace(',', '') == dd[0].replace(',', '')) and \
        #                 (et[1].replace(',', '') == dd[1].replace(',', '')) and \
        #                 (et[2].replace(',', '') == dd[2].replace(',', '')) \
        #                 or
        #                 (('<' + et[0].lower() + '>' == dd[0].lower()) and
        #                  ('<' + et[1].lower() + '>' == dd[1].lower()) and
        #                  ('<' + et[2].lower() + '>' == dd[2].lower()))):
        #             embeddings_test_final[i] = et
        #             i = i + 1
        #             break
        # i = 0
        # for dd in valid_set:
        #     # print(dd)
        #     for et in embeddings_valid.values():
        #         if ((et[0].replace(',', '') == dd[0].replace(',', '')) and\
        #                 (et[1].replace(',', '') == dd[1].replace(',', '')) and\
        #                 (et[2].replace(',', '') == dd[2].replace(',', ''))
        #                 or
        #                 (('<' + et[0].lower() + '>' == dd[0].lower()) and
        #                  ('<' + et[1].lower() + '>' == dd[1].lower()) and
        #                  ('<' + et[2].lower() + '>' == dd[2].lower()))):
        #             embeddings_valid_final[i] = et
        #             i = i + 1
        #             break
        if (len(embeddings_valid)!= len(valid_set)) and (len(embeddings_test)!= len(test_set)):
            print("check lengths of valid and test data:valid_emb:"+str(len(embeddings_valid))+" valid_set"+str(len(valid_set))+"test_set:"+str(len(test_set))+"test_emb:"+str(len(embeddings_test)))
            # exit(1)
        train_i = 0
        test_set_copy = deepcopy(test_set)
        valid_set_copy = deepcopy(valid_set)
        embeddings_test_final = dict()
        embeddings_valid_final = dict()
        for dd in test_set:
            found_data = False
            jj = 0
            for sd in embeddings_test.values():
                sub = self.update_entity(self, dd[0])
                pred = self.update_entity(self, dd[1])
                obj = self.update_entity(self, dd[2])
                sub1 = '<' + self.update_entity(self, sd[0]) + '>'
                pred1 = '<' + self.update_entity(self, sd[1]) + '>'
                obj1 = '<' + self.update_entity(self, sd[2]) + '>'
                if (((sub == sub1) and (pred == pred1) and (obj == obj1))
                        or
                        ((sub1.lower() == sub.lower()) and (pred1.lower() == pred.lower()) and
                         (obj1.lower() == obj.lower()))):
                    embeddings_test_final[train_i] = sd
                    train_i += 1
                    found_data = True
                    break
                jj += 1
            if found_data == False:
                test_set_copy.remove(dd)
                print("missing test data from sentence embeddings file:" + str(dd))
            else:
                # embeddings_test.pop(jj)
                embeddings_test = self.without(embeddings_test, jj)
                # print("to delete from list: " + str(sd))
                # del embeddings_test[sd]

        test_set = deepcopy(test_set_copy)

        train_i = 0
        for dd in valid_set:
            found_data = False
            jj = 0
            for sd in embeddings_valid.values():
                sub = self.update_entity(self, dd[0])
                pred = self.update_entity(self, dd[1])
                obj = self.update_entity(self, dd[2])
                sub1 = '<' + self.update_entity(self, sd[0]) + '>'
                pred1 = '<' + self.update_entity(self, sd[1]) + '>'
                obj1 = '<' + self.update_entity(self, sd[2]) + '>'
                if (((sub == sub1) and (pred == pred1) and (obj == obj1))
                        or
                        ((sub1.lower() == sub.lower()) and (pred1.lower() == pred.lower()) and
                         (obj1.lower() == obj.lower()))):
                    embeddings_valid_final[train_i] = sd
                    train_i += 1
                    found_data = True
                    break
                jj += 1
            if found_data == False:
                valid_set_copy.remove(dd)
                print("missing valid data from sentence embeddings file:" + str(dd))
            else:
                # print("to delete from list: " + str(sd))
                # embeddings_valid.pop(jj)
                embeddings_valid = self.without(embeddings_valid, jj)


        valid_set = deepcopy(valid_set_copy)

        return embeddings_test_final.values(), embeddings_valid_final.values(), test_set, valid_set


        # return embeddings.values()


# args = argparse_default()
# dataset = Data(args=args)

# # Test data class
# bpdp = True
# if not bpdp:
#     properties_split = ["deathPlace/","birthPlace/","author/","award/","foundationPlace/","spouse/","starring/","subsidiary/"]
#     datasets_class = ["range/","domain/","mix/","property/","domainrange/","random/"]
#     # make it true or false
#     prop_split = True
#     clss = datasets_class
#     if prop_split:
#         clss = properties_split
#
#     for cls in clss:
#         method = "emb-only" #emb-only  hybrid
#         path_dataset_folder = 'dataset/'
#         if prop_split:
#             dataset = Data(data_dir=path_dataset_folder, sub_dataset_path= None, prop = cls)
#         else:
#             dataset = Data(data_dir=path_dataset_folder, sub_dataset_path= cls)
# else:
#     path_dataset_folder = 'dataset/hybrid_data/bpdp/'
#     dataset = Data(data_dir=path_dataset_folder, bpdp_dataset=True)
#     print("success")
