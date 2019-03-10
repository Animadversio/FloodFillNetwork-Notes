#%%
import numpy as np
from ffn.utils.proofreading import GraphUpdater, ObjectReview
import networkx as nx
from neuroglancer_segment_visualize import GraphUpdater_show

import pickle

#%%
from ffn.utils.proofreading import Base
import copy
import collections
import neuroglancer
class ManualAgglomeration(Base):
    """
    """
    def set_init_state(self):
        pass

    def __init__(self, graph, viewer, objects=None):
        """Constructor.
        Args:
          graph: networkx Graph object to modify
          objects: iterable of object IDs or iterables of object IDs
          bad: set in which to store objects flagged as bad
        """
        super(ManualAgglomeration, self).__init__()
        # self.viewer.set_state(copy.deepcopy(viewer.state))
        self.viewer = viewer
        print(self.viewer.get_viewer_url())
        self.graph = graph
        self.split_objects = []
        self.split_path = []
        self.split_index = 1
        if objects == None:
            self.objects = dict()
        else:
            for base_id, id_list in objects.items():
                assert type(base_id) is int and type(id_list) is list
            self.objects = objects
        self.cur_base_id = None
        self.cur_obj_id = None
        self.cur_id_list = [] # maintain as list (Maybe a set is better! )
        self.realtime_merge = True
        self.realtime_mod_edge = True

        self.viewer.actions.add('set-base-seg', lambda s: self.set_base_segment(s))
        self.viewer.actions.add('add-segment-to-group', lambda s: self.add_segment_to_group(s))
        self.viewer.actions.add('isolate-segment-from-group', lambda s: self.isolate_segment_from_group(s))
        self.viewer.actions.add('add-ccs', lambda s: self.add_ccs())
        self.viewer.actions.add('merge-segments', lambda s: self.merge_segments())
        self.viewer.actions.add('visualize-cc', lambda s: self.visualize_cc())
        self.viewer.actions.add('visualize-objs', lambda s: self.visualize_objects())
        self.viewer.actions.add('save-objects', lambda s: self.save_objects())
        self.viewer.actions.add('clear-equiv', lambda s: self.clear_equiv())
        self.viewer.actions.add('toggle-realtime-merge', lambda s: self.toggle_realtime_merge())
        # self.viewer.actions.add('clear-splits', lambda s: self.clear_splits())
        # self.viewer.actions.add('add-split', self.add_split)
        # self.viewer.actions.add('accept-split', lambda s: self.accept_split())
        # self.viewer.actions.add('split-inc', lambda s: self.inc_split())
        # self.viewer.actions.add('split-dec', lambda s: self.dec_split())
        # self.viewer.actions.add('mark-bad', lambda s: self.mark_bad())
        # self.viewer.actions.add('next-batch', lambda s: self.next_batch())
        # self.viewer.actions.add('prev-batch', lambda s: self.prev_batch())
        # self.viewer.actions.add('toggle-equiv', lambda s: self.toggle_equiv())

        with self.viewer.config_state.txn() as s:
            s.input_event_bindings.data_view['shift+mousedown2'] = 'set-base-seg'
            s.input_event_bindings.data_view['alt+mousedown0'] = 'add-segment-to-group'
            s.input_event_bindings.data_view['alt+mousedown2'] = 'isolate-segment-from-group'
            s.input_event_bindings.viewer['keyt'] = 'toggle-realtime-merge'
            s.input_event_bindings.viewer['keyv'] = 'visualize-objs'
            s.input_event_bindings.viewer['keym'] = 'merge-segments'
            s.input_event_bindings.viewer['keys'] = 'save-objects'
            s.input_event_bindings.viewer['keyc'] = 'add-ccs'
            s.input_event_bindings.viewer['keyd'] = 'visualize-cc'
            s.input_event_bindings.viewer['keyx'] = 'clear-equiv'
            # s.input_event_bindings.viewer['keyj'] = 'next-batch'
            # s.input_event_bindings.viewer['keyk'] = 'prev-batch'
            # s.input_event_bindings.viewer['keya'] = 'clear-splits'
            # s.input_event_bindings.viewer['bracketleft'] = 'split-dec'
            # s.input_event_bindings.viewer['bracketright'] = 'split-inc'
            # s.input_event_bindings.viewer['keys'] = 'accept-split'
            # s.input_event_bindings.data_view['shift+mousedown0'] = 'add-split'
            # s.input_event_bindings.viewer['keyv'] = 'mark-bad'
            # s.input_event_bindings.viewer['keyt'] = 'toggle-equiv'
        self.apply_equivs = True
        with self.viewer.txn() as s:
            s.layers['orig'] = neuroglancer.SegmentationLayer(
                source=s.layers['seg'].source)
            s.layers['orig'].visible = False
            s.layers['orig'].selectedAlpha = 0.15

    def set_base_segment(self, action_state):
        segment_id = action_state.selected_values.get('seg')
        print(action_state.selected_values)
        print(segment_id)
        if segment_id is None: return
        elif not np.isscalar(segment_id):  # if the id has already belongs to some segments, then it's a map
            assert type(segment_id) is neuroglancer.MapEntry
            segment_id=segment_id.key
        if not self.cur_base_id == segment_id:
            if self.cur_base_id is not None:
                self.objects[self.cur_base_id] = self.cur_id_list
            self.cur_obj_id = segment_id # current selected id
            if np.any([segment_id in id_list for id_list in self.objects.values()]):
                for base_id, id_list in self.objects.items():
                    if segment_id in id_list:
                        self.cur_base_id = base_id
                        self.cur_id_list = id_list
                        break
                self.update_msg('Set base id=%d (segment %d belongs to). current id_list %s' % (self.cur_base_id, segment_id,
                                ','.join([str(idx) for idx in self.cur_id_list])))
            else:
                self.cur_base_id = segment_id  # select current base segment
                self.objects[segment_id] = [segment_id]
                self.cur_id_list = [segment_id]
                self.update_msg('Set base id=%d. current id_list %s' % (segment_id,
                                ','.join([str(idx) for idx in self.cur_id_list])))
        else:
            self.objects[segment_id] = self.cur_id_list
            self.update_msg('Dissociate from base id= %d. Save id_list %s' % (segment_id,
                                ','.join([str(idx) for idx in self.cur_id_list])))
            self.cur_base_id = None
            self.cur_id_list = []

    def add_segment_to_group(self, action_state):
        segment_id = action_state.selected_values.get('seg')  # use the seg id to merge item
        print(action_state.selected_values)
        print(segment_id)
        if segment_id is None: return
        elif not np.isscalar(segment_id):  # if the id has already belongs to some segments, then it's a map
            assert type(segment_id) is neuroglancer.MapEntry
            group_map_id = segment_id.value
            segment_id = segment_id.key
            print("Map %d to %d"%(segment_id, group_map_id ))
        if not type(segment_id) is int:
            print("Type error, type=", type(segment_id))
            return
        if self.cur_base_id == None:
            return
        elif segment_id == 0:
            self.update_msg("Background segment 0 cannot be added!")
            return
        else:
            old_id = self.cur_obj_id
            self.cur_obj_id = segment_id  # change the seed segment
            print("Current object id:", self.cur_obj_id)
            if segment_id not in self.cur_id_list:
                # supervoxel out of current group
                if np.any([segment_id in id_list for id_list in self.objects.values()]):
                    # the thing to merge is included in other segment groups
                    for base_id, id_list in self.objects.items():
                        if segment_id in id_list:
                            base_id_to_merge, id_list_to_merge = base_id, id_list
                            break
                    self.update_msg('Going to merge up object %d and %d. %d is going to be deleted. \n'
                                    'Selected segment %d with (%s) going to Object %d. \nCurrent id_list %s' % (
                        self.cur_base_id, base_id_to_merge, base_id_to_merge,
                        segment_id, ','.join([str(idx) for idx in id_list_to_merge]), self.cur_base_id,
                        ','.join([str(idx) for idx in self.cur_id_list])))
                    if input("Going to merge object %d and %d. %d will be deleted. Influenced supervoxels id=%s \n Input [Y] to continue."
                             %(self.cur_base_id, base_id_to_merge, base_id_to_merge,
                               ','.join([str(idx) for idx in self.cur_id_list])) )=='Y':
                        self.objects.pop(base_id_to_merge)
                        self.cur_id_list.extend(id_list_to_merge)
                        self.update_msg('Merged up object %d and %d. %d has been deleted. \n'
                                        'Selected segment %d with (%s) added to Object %d. \nCurrent id_list %s' % (
                            self.cur_base_id, base_id_to_merge, base_id_to_merge,
                            segment_id, ','.join([str(idx) for idx in id_list_to_merge]), self.cur_base_id,
                            ','.join([str(idx) for idx in self.cur_id_list])))
                    else:
                        self.update_msg('Operation aborted\nSelected segment %d added to Object %d. \nCurrent id_list %s' % (
                        segment_id, self.cur_base_id,
                        ','.join([str(idx) for idx in self.cur_id_list])))
                        return
                else:
                    # the supervoxel is an isolated one
                    if not segment_id ==0 :
                        self.cur_id_list.append(segment_id)
                        self.update_msg('Selected segment %d added to Object %d. \nCurrent id_list %s' % (segment_id, self.cur_base_id,
                                        ','.join([str(idx) for idx in self.cur_id_list])))
                if self.realtime_mod_edge:
                    self.graph.add_edge(segment_id, old_id)
            else:
                self.update_msg('Selected segment %d is already in Object %d. \nCurrent id_list %s' % (segment_id, self.cur_base_id,
                                ','.join([str(idx) for idx in self.cur_id_list])))
            if self.realtime_merge:
                self.visualize_objects()

    def isolate_segment_from_group(self, action_state):
        segment_id = action_state.selected_values.get('orig') # get the id from original segmentation! or it will get merged id
        print(action_state.selected_values)
        print(segment_id)
        if segment_id is None: return
        if self.cur_base_id == None:
            return
        else:
            if segment_id in self.cur_id_list:
                self.cur_id_list.remove(segment_id)
                self.update_msg('Selected segment %d removed from Object %d. \nCurrent id_list %s' % (segment_id, self.cur_base_id,
                                ','.join([str(idx) for idx in self.cur_id_list])))
                rm_edge_list = []
                if self.realtime_mod_edge:
                    for idx in self.graph.neighbors(segment_id):
                        if idx in self.cur_id_list: rm_edge_list.append((idx, segment_id))  # incase the size of the neighbors change
                    self.graph.remove_edges_from(rm_edge_list)
            else:
                self.update_msg("Object %d includes no segment %d !" % (self.cur_base_id, segment_id))
        if self.realtime_merge:
            self.visualize_objects()

    def save_objects(self):
        if self.cur_base_id is not None:
            self.objects[self.cur_base_id] = self.cur_id_list
        print(self.objects)

    def visualize_objects(self):
        if self.cur_base_id is not None:
            self.objects[self.cur_base_id] = self.cur_id_list
        s = copy.deepcopy(self.viewer.state)
        l = s.layers['seg']
        l.equivalences.clear()
        for base_id, id_list in self.objects.items():
            l.equivalences.union(*id_list) # note cannot use list here
        self.viewer.set_state(s)

    def merge_segments(self):
        sids = self.cur_id_list  # [sid for sid in self.viewer.state.layers['seg'].segments if sid > 0]
        self.graph.add_edges_from(zip(sids, sids[1:]))
        # l.equivalences.union(*sids)

    def toggle_realtime_merge(self):
        if self.realtime_merge:
            self.realtime_merge = False
            self.update_msg("Turn off real time merge.")
        else:
            self.realtime_merge = True
            self.visualize_objects()
            self.update_msg("Turn on real time merge.")

    def clear_equiv(self):
        s = copy.deepcopy(self.viewer.state)
        l = s.layers['seg']
        l.equivalences.clear()
        self.viewer.set_state(s)

    def visualize_cc(self):
        # add this to ease the manual merging stuff
        if self.apply_equivs:
            s = copy.deepcopy(self.viewer.state)
            l = s.layers['seg']
            for idx in l.segments:
                cc = nx.node_connected_component(self.graph, idx)
                print(str(cc))
                l.equivalences.union(*cc)
            self.viewer.set_state(s)

    def add_ccs(self):
        curr = set(self.viewer.state.layers['seg'].segments)
        for sid in self.viewer.state.layers['seg'].segments:
            if sid in self.graph:
                curr |= set(nx.node_connected_component(self.graph, sid))

        self.update_segments(curr)

    def export_merge_data(self):
        # np.savez("agglomeration_save.npz",objects=self.objects, graph=self.graph)
        pickle.dump({"objects": self.objects, "graph": self.graph}, open("p11_agglomeration.pkl", "wb"))
        return self.objects, self.graph
#%%
from neuroglancer_segment_visualize import neuroglancer_visualize
graph = nx.Graph()
seg = np.zeros((100,100,100),dtype=np.uint32)
seg[:50,:50,:20]=1
seg[:50,50:,:30]=2
seg[50:,:50,5:45]=3
seg[50:,50:,:50]=4
seg[:50,:50,40:80]=5
seg[:50,50:,30:]=6
seg[50:,:50,50:]=7
seg[50:,50:,60:]=8
objects = np.unique(seg,)
if objects[0] == 0:
    objects = objects[1:]
graph.add_nodes_from(objects)
# graph_update = GraphUpdater_show(graph, [], [], {'seg': {"vol": seg}, }, None)

#%%
viewer = neuroglancer_visualize({'seg':{'vol':seg}}, None)
agg_tool = ManualAgglomeration(graph, viewer)
#%%
import networkx as nx
from ffn.inference.storage import subvolume_path
from neuroglancer_segment_visualize import neuroglancer_visualize
p = pickle.load(open("p11_agglomeration.pkl","rb"))
objects, graph = p['objects'], p['graph']
# graph = nx.Graph()
# # seg = np.load(subvolume_path("/home/morganlab/Documents/ixP11LGN/p11_6_consensus_33_38_full/", (0, 0, 0), "npz"))
seg = np.load(subvolume_path("/Users/binxu/Connectomics_Code/results/LGN/p11_6_consensus_33_38_full/", (0, 0, 0), "npz"))
segmentation = seg["segmentation"]
seg.close()
# objects = np.unique(segmentation,)
# # objects, cnts = np.unique(segmentation, return_counts=True)
# # objects = objects
# #image_dir = "/home/morganlab/Documents/ixP11LGN/EM_data/p11_6_EM/grayscale_ixP11_6_align_norm.h5"
# graph.add_nodes_from(objects[1:])
viewer = neuroglancer_visualize({'seg': {"vol": segmentation}, }, None)
# #%%
agg_tool = ManualAgglomeration(graph, viewer, objects)
#%%
objects, graph = agg_tool.objects, agg_tool.graph
pickle.dump({"objects":objects, "graph": graph}, open("p11_agglomeration.pkl","wb"))
#%%
