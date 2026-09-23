#!/usr/bin/env python3
"""Read-only ROS probe for Ghost Game Detection2DArray; never commands the arm."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time


def stats(values):
    if not values:
        return None
    values = sorted(values)
    def quantile(q):
        x = q*(len(values)-1)
        low = int(x)
        high = min(low+1,len(values)-1)
        return values[low]+(values[high]-values[low])*(x-low)
    return dict(count=len(values),p50=quantile(.5),p95=quantile(.95),max=values[-1])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds',type=float,default=60.)
    parser.add_argument('--topic',default='/nearest_face/tracking')
    parser.add_argument('--out',default='face-follow-latency.json')
    args,ros_args=parser.parse_known_args()
    if not 0 < args.seconds <= 600:
        parser.error('--seconds must be in (0,600]')
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile,ReliabilityPolicy
    from vision_msgs.msg import Detection2DArray
    from std_msgs.msg import String

    class Probe(Node):
        def __init__(self):
            super().__init__('face_follow_latency_probe')
            self.rows,self.metrics,self.control=[],[],[]
            qos=QoSProfile(depth=1,reliability=ReliabilityPolicy.BEST_EFFORT)
            self.sub=self.create_subscription(Detection2DArray,args.topic,self.receive,qos)
            self.diag=self.create_subscription(String,'/nearest_face/latency',lambda m:self.record_json(m,self.metrics),qos)
            self.state=self.create_subscription(String,'/ghost_game_node/state',self.receive_state,qos)

        def record_json(self,message,rows):
            try: rows.append(json.loads(message.data))
            except (ValueError,TypeError): pass

        def receive_state(self,message):
            try:
                value=json.loads(message.data)
                self.control.append(dict(phase=value.get('phase'),face=value.get('face')))
            except (ValueError,TypeError): pass

        def receive(self,message):
            stamp=message.header.stamp.sec*1_000_000_000+message.header.stamp.nanosec
            now=self.get_clock().now().nanoseconds
            self.rows.append(dict(arrival=time.perf_counter(),stamp_ns=stamp,
                                  age_ms=(now-stamp)/1e6 if stamp else None,
                                  valid=bool(message.detections),
                                  target_id=message.detections[0].id if message.detections else ''))

    rclpy.init(args=ros_args)
    node=Probe()
    started=time.perf_counter()
    try:
        while rclpy.ok() and time.perf_counter()-started < args.seconds:
            rclpy.spin_once(node,timeout_sec=.05)
    except KeyboardInterrupt:
        pass
    finally:
        duration=time.perf_counter()-started
        valid=[row for row in node.rows if row['valid']]
        report=dict(duration_seconds=duration,topic=args.topic,
                    note='Read-only best-effort observer may drop messages. Source age requires synchronized ROS clocks. This is not physical arm response time. No images are collected.',
                    valid_hz=len(valid)/max(duration,1e-9),messages=len(node.rows),
                    valid_source_age_ms=stats([r['age_ms'] for r in valid if r['age_ms'] is not None]),
                    valid_intervals_ms=stats([(b['arrival']-a['arrival'])*1000 for a,b in zip(valid,valid[1:])]),
                    target_ids=dict(Counter(r['target_id'] for r in valid)),
                    rows=node.rows,perception_metrics=node.metrics,control=node.control)
        out=Path(args.out)
        out.parent.mkdir(parents=True,exist_ok=True)
        out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps({k:report[k] for k in ['valid_hz','messages','valid_source_age_ms','valid_intervals_ms']},indent=2))
        print('Saved',out)
        node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()


if __name__=='__main__':
    main()
