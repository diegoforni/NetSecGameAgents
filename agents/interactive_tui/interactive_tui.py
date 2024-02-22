from textual.app import App, ComposeResult, Widget
from textual.widgets import Tree, Button, Log, Select, Input
from textual.containers import Vertical, VerticalScroll
from textual.validation import Function
from textual import on

import sys
from os import path
import os
import logging
import ipaddress
from random import choice
import argparse

# This is used so the agent can see the environment and game components
sys.path.append(
    path.dirname(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))
)
from env.game_components import Network, IP
from env.game_components import ActionType, Action, GameState, Observation

# This is used so the agent can see the BaseAgent
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from base_agent import BaseAgent

log_filename = os.path.dirname(os.path.abspath(__file__)) + "/interactive_tui_agent.log"
logging.basicConfig(
    filename=log_filename,
    filemode="w",
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.DEBUG,
)
logger = logging.getLogger("Interactive-TUI-agent")
logger.info("Start")


def is_valid_ip(ip_addr: str) -> bool:
    """Validate if the input string is an IPv4 address"""
    try:
        ipaddress.IPv4Address(ip_addr)
        return True
    except ipaddress.AddressValueError:
        return False


def is_valid_net(net_addr: str) -> bool:
    """Validate if the input string is an IPv4 or IPv6 network"""
    try:
        ipaddress.ip_network(net_addr)
        return True
    except (ipaddress.NetmaskValueError, ipaddress.AddressValueError, ValueError):
        return False


class TreeState(Widget):
    def __init__(self, obs: Observation):
        super().__init__()
        self.init_obs = obs

    def _create_tree_from_obs(self, observation: Observation) -> Tree:
        tree: Tree[dict] = Tree("State", classes="box")
        tree.root.expand()
        state = observation.state

        contr_host_list = [host for host in state.controlled_hosts]
        known_host_list = [
            host for host in state.known_hosts if host not in contr_host_list
        ]

        networks = tree.root.add("Known Networks", expand=True)
        for network in state.known_networks:
            networks.add(str(network))

        known_hosts = tree.root.add("Known Hosts", expand=True)
        for host in known_host_list:
            h = known_hosts.add(str(host))
            for s_host in state.known_services:
                if s_host == host:
                    node = h.add("Services", expand=True)
                    for service in state.known_services[s_host]:
                        node.add_leaf(service.name)

        owned_hosts = tree.root.add("Controlled Hosts", expand=True)
        for host in contr_host_list:
            h = owned_hosts.add(str(host))

            # Add any services
            for s_host in state.known_services:
                if s_host == host:
                    node = h.add("Services", expand=True)
                    for service in state.known_services[s_host]:
                        node.add_leaf(service.name)
            # Add known data
            for d_host in state.known_data:
                if d_host == host:
                    node = h.add("Data", expand=True)
                    for datum in state.known_data[host]:
                        node.add_leaf(f"{datum.owner} - {datum.id}")

        return tree

    def compose(self) -> ComposeResult:
        tree = self._create_tree_from_obs(self.init_obs)
        yield tree


class InteractiveTUI(App):
    """App to display key events."""

    CSS_PATH = "layout.tcss"

    def __init__(self, host: str, port: int, role: str, mode: str):
        super().__init__()
        self.returns = 0
        self.next_action = None
        self.src_host_input = ""
        self.target_host_input = ""
        self.network_input = ""
        self.service_input = ""
        self.data_input = ""
        self.agent = BaseAgent(host, port, role)
        self.current_obs = self.agent.register()
        self.mode = mode

    def compose(self) -> ComposeResult:
        yield Vertical(TreeState(obs=self.current_obs), classes="box", id="tree")
        yield Select(
            [
                ("ScanNetwork", ActionType.ScanNetwork),
                ("ScanServices", ActionType.FindServices),
                ("ExploitService", ActionType.ExploitService),
                ("FindData", ActionType.FindData),
                ("ExfiltrateData", ActionType.ExfiltrateData),
            ],
            prompt="Select Action",
            classes="box",
        )
        if self.mode == "guided":
            yield Vertical(
                VerticalScroll(Select([], prompt="Select source host", id="src_host")),
                VerticalScroll(Select([], prompt="Select network", id="network")),
                VerticalScroll(
                    Select(
                        [],
                        prompt="Select target host",
                        id="target_host",
                    )
                ),
                VerticalScroll(Select([], prompt="Select service", id="service")),
                VerticalScroll(Select([], prompt="Select data", id="data")),
                classes="params",
            )
        else:
            yield Vertical(
                VerticalScroll(
                    Input(
                        placeholder="Source Host",
                        id="src_host",
                        validators=[Function(is_valid_ip, "This is not a valid IP.")],
                    ),
                    Input(
                        placeholder="Network",
                        id="network",
                        validators=[
                            Function(is_valid_net, "This is not a valid Network.")
                        ],
                    ),
                    Input(
                        placeholder="Target Host",
                        id="target_host",
                        validators=[Function(is_valid_ip, "This is not a valid IP.")],
                    ),
                    Input(placeholder="Service", id="service"),
                    Input(placeholder="Data", id="data"),
                    classes="box",
                )
            )
        yield Log(classes="box", id="textarea")
        yield Button("Take Action", variant="primary")
        # yield Footer()

    @on(Select.Changed)
    def select_changed(self, event: Select.Changed) -> None:
        log = self.query_one("Log")
        log.write_line(str(event.value))

        # Save the selections of the action parameters
        match event._sender.id:
            case "src_host":
                self.src_host_input = event.value
                return
            case "network":
                self.network_input = event.value
                return
            case "target_host":
                self.target_host_input = event.value
                return
            case "service":
                self.service_input = event.value
                return
            case "data":
                self.data_input = event.value
                return
            case _:
                # otherwise it is the action selector
                self.next_action = event.value

        state = self.current_obs.state

        if self.mode == "guided":
            net_input = self.query_one("#network", Select)
            target_input = self.query_one("#target_host", Select)
            service_input = self.query_one("#service", Select)
            data_input = self.query_one("#data", Select)

            contr_hosts = [(str(host), str(host)) for host in state.controlled_hosts]
            src_input = self.query_one("#src_host", Select)
            src_input.set_options(contr_hosts)

            known_hosts = [
                (str(host), str(host))
                for host in state.known_hosts
                if host not in state.controlled_hosts
            ]

            match event.value:
                case ActionType.ScanNetwork:
                    networks = [
                        (str(net), str(net))
                        for net in self.current_obs.state.known_networks
                    ]
                    net_input.set_options(networks)
                case ActionType.FindServices:
                    target_input.set_options(known_hosts)
                case ActionType.ExploitService:
                    target_input.set_options(known_hosts)

                    services = []
                    for host in state.known_services:
                        if host not in state.controlled_hosts:
                            for serv in state.known_services[host]:
                                services.append((serv.name, serv.name))
                    service_input.set_options(services)

                case ActionType.FindData:
                    target_input.set_options(contr_hosts)

                case ActionType.ExfiltrateData:
                    target_input.set_options(contr_hosts)

                    data = []
                    for host in state.known_data:
                        for d in state.known_data[host]:
                            data.append((d.id, d.id))
                    data_input.set_options(data)

        else:
            net_input = self.query_one("#network", Input)
            target_input = self.query_one("#target_host", Input)
            service_input = self.query_one("#service", Input)
            data_input = self.query_one("#data", Input)

        # Set proper visibility
        if event.value == ActionType.ScanNetwork:
            net_input.visible = True
            service_input.visible = False
            target_input.visible = False
            data_input.visible = False
        elif event.value == ActionType.FindServices:
            net_input.visible = False
            service_input.visible = False
            target_input.visible = True
            data_input.visible = False
        elif event.value == ActionType.ExploitService:
            net_input.visible = False
            service_input.visible = True
            target_input.visible = True
            data_input.visible = False
        elif event.value == ActionType.FindData:
            net_input.visible = False
            service_input.visible = False
            target_input.visible = True
            data_input.visible = False
        elif event.value == ActionType.ExfiltrateData:
            net_input.visible = False
            service_input.visible = False
            target_input.visible = True
            data_input.visible = True
        else:
            net_input.visible = True
            service_input.visible = True
            target_input.visible = True
            data_input.visible = True

    @on(Input.Changed)
    def handle_inputs(self, event: Input.Changed) -> None:
        # log = self.query_one("Log")
        # log.write_line(f"Input: {str(event.value)} from {event._sender.id}")

        if event._sender.id == "src_host":
            self.src_host_input = event.value
        elif event._sender.id == "network":
            self.network_input = event.value
        elif event._sender.id == "target_host":
            self.target_host_input = event.value
        elif event._sender.id == "service":
            self.service_input = event.value
        elif event._sender.id == "data":
            self.data_input = event.value

    @on(Button.Pressed)
    def submit_action(self, event: Button.Pressed) -> None:
        """
        Press the button to select a random action.
        Right now there is only one button. If we add more we will need to distinguish them.
        """
        self.update_state()

        # Take the first node of TreeState which contains the tree
        tree_state = self.query_one(TreeState)
        tree = tree_state.children[0]
        self.update_tree(tree)

    def update_state(self) -> None:
        action = self._move(self.current_obs.state)
        # Get next observation of the environment
        next_observation = self.agent.make_step(action)
        # Collect reward
        self.returns += next_observation.reward
        # Move to next state
        self.current_obs = next_observation

        if next_observation.end:
            self.notify(f"You won! Total return: {self.returns}", timeout=10)
            self._clear_state()

    def update_tree(self, tree: Widget) -> None:
        """Update the tree with the new state"""

        # Get the new state
        new_state = self.current_obs.state

        # First remove all the children and then rebuild the tree
        # This is faster than looking at the delta
        tree.root.remove_children()

        contr_host_list = new_state.controlled_hosts
        known_host_list = [
            host for host in new_state.known_hosts if host not in contr_host_list
        ]

        networks = tree.root.add("Known Networks", expand=True)
        for network in new_state.known_networks:
            networks.add(str(network))

        known_hosts = tree.root.add("Known Hosts", expand=True)
        for host in known_host_list:
            h = known_hosts.add(str(host), expand=True)
            for s_host in new_state.known_services:
                if s_host == host:
                    node = h.add("Services", expand=True)
                    for service in new_state.known_services[s_host]:
                        node.add_leaf(service.name)

        owned_hosts = tree.root.add("Controlled Hosts", expand=True)
        for host in contr_host_list:
            h = owned_hosts.add(str(host), expand=True)

            # Add any services
            for s_host in new_state.known_services:
                if s_host == host:
                    node = h.add("Services", expand=True)
                    for service in new_state.known_services[s_host]:
                        node.add_leaf(service.name)
            # Add known data
            for d_host in new_state.known_data:
                if d_host == host:
                    node = h.add("Data", expand=True)
                    for datum in new_state.known_data[host]:
                        node.add_leaf(f"{datum.owner} - {datum.id}")

    def _generate_valid_actions(self, state: GameState) -> list:
        # Generate the list of all valid actions in the current state
        valid_actions = set()
        for src_host in state.controlled_hosts:
            # Network Scans
            for network in state.known_networks:
                valid_actions.add(
                    Action(
                        ActionType.ScanNetwork,
                        params={"target_network": network, "source_host": src_host},
                    )
                )
            # Service Scans
            for host in state.known_hosts:
                valid_actions.add(
                    Action(
                        ActionType.FindServices,
                        params={"target_host": host, "source_host": src_host},
                    )
                )
            # Service Exploits
            for host, service_list in state.known_services.items():
                for service in service_list:
                    valid_actions.add(
                        Action(
                            ActionType.ExploitService,
                            params={
                                "target_host": host,
                                "target_service": service,
                                "source_host": src_host,
                            },
                        )
                    )
            # Data Scans
            for host in state.controlled_hosts:
                valid_actions.add(
                    Action(
                        ActionType.FindData,
                        params={"target_host": host, "source_host": src_host},
                    )
                )

        # Data Exfiltration
        for src_host, data_list in state.known_data.items():
            for data in data_list:
                for trg_host in state.controlled_hosts:
                    if trg_host != src_host:
                        valid_actions.add(
                            Action(
                                ActionType.ExfiltrateData,
                                params={
                                    "target_host": trg_host,
                                    "source_host": src_host,
                                    "data": data,
                                },
                            )
                        )
        return list(valid_actions)

    def _move(self, state: GameState) -> Action:
        action = None
        log = self.query_one("Log")
        if self.next_action == ActionType.ScanNetwork:
            parameters = {
                "source_host": IP(self.src_host_input),
                "target_network": Network(
                    self.network_input[:-3], mask=int(self.network_input[-2:])
                ),
            }
            action = Action(action_type=self.next_action, params=parameters)
        elif self.next_action in [ActionType.FindServices, ActionType.FindData]:
            parameters = {
                "source_host": IP(self.src_host_input),
                "target_host": IP(self.target_host_input),
            }
            action = Action(action_type=self.next_action, params=parameters)
        elif self.next_action == ActionType.ExploitService:
            for host, service_list in state.known_services.items():
                if IP(self.target_host_input) == host:
                    for service in service_list:
                        if self.service_input == service.name:
                            parameters = {
                                "source_host": IP(self.src_host_input),
                                "target_host": IP(self.target_host_input),
                                "target_service": service,
                            }
                            action = Action(
                                action_type=self.next_action, params=parameters
                            )
        elif self.next_action == ActionType.ExfiltrateData:
            for host, data_items in state.known_data.items():
                log.write_line(f"{str(state.known_data.items())}")
                if IP(self.src_host_input) == host:
                    for datum in data_items:
                        if self.data_input == datum.id:
                            parameters = {
                                "source_host": IP(self.src_host_input),
                                "target_host": IP(self.target_host_input),
                                "data": datum,
                            }
                            action = Action(
                                action_type=self.next_action, params=parameters
                            )
        else:
            log.write_line(f"Invalid input: {self.next_action} with {parameters}")
            logger.info(
                f"Invalid input from user: {self.next_action} with {parameters}"
            )

        if action is None:
            action = self._random_move(state)
            log.write_line(f"Random action: {str(action)}")
            logger.info(f"Random action due to error: {str(action)}")

        log.write_line(f"Action to take: {str(action)}")
        logger.info(f"User selected action: {str(action)}")

        return action

    def _random_move(self, state: GameState) -> Action:
        # Randomly choose from the available actions
        valid_actions = self._generate_valid_actions(state)
        # if self.args.force_ignore:
        #     valid_actions = [action for action in valid_actions if action not in actions_taken]
        return choice(valid_actions)

    def _clear_state(self) -> None:
        """Reset the state and variables"""
        logger.info(f"Reset the environment and state")
        self.current_obs = self.agent.request_game_reset()
        self.next_action = None
        self.src_host_input = ""
        self.target_host_input = ""
        self.network_input = ""
        self.service_input = ""
        self.data_input = ""

        selectors = self.query(Select)
        for selector in selectors:
            selector.clear()

        for inp in self.query(Input):
            inp.clear()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--task_config_file", help="Reads the task definition from a configuration file", default=path.join(path.dirname(__file__), 'netsecenv-task.yaml'), action='store', required=False)
    parser.add_argument(
        "--host",
        help="Host where the game server is",
        default="127.0.0.1",
        action="store",
        required=False,
    )
    parser.add_argument(
        "--port",
        help="Port where the game server is",
        default=9000,
        type=int,
        action="store",
        required=False,
    )
    parser.add_argument(
        "--role", help="Role of the agent", default="Attacker", choices=["Attacker"]
    )
    parser.add_argument(
        "--mode", type=str, choices=["guided", "normal"], default="normal"
    )
    args = parser.parse_args()

    logger.info("Creating the agent")
    app = InteractiveTUI(args.host, args.port, args.role, args.mode)
    app.run()